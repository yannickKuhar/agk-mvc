"""
solver/mvc_solver.py
--------------------
Minimum Vertex Cover solver with two backends:

  1. ILP (exact) via PuLP + CBC solver
     — Finds the provably optimal MVC.
     — Time complexity: exponential in worst case, but fast in practice
       on reduced graphs.

  2. NetworkX 2-approximation (fallback)
     — Uses maximum matching + König's theorem for bipartite graphs,
       or a greedy 2-approximation for general graphs.
     — Used when the graph is too large for ILP or PuLP is unavailable.

Usage:
    solver = MVCSolver(backend='ilp', timeout=60)
    result = solver.solve(G)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Set

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Solver availability check (run once at import time)
# ---------------------------------------------------------------------------

# Priority order for 'auto' selection (fastest first)
_SOLVER_PRIORITY = ["gurobi", "xpress", "highs", "scip", "cbc", "glpk"]
_AVAILABLE_SOLVERS: dict = {}


def _check_solvers() -> None:
    global _AVAILABLE_SOLVERS
    try:
        import pulp
        candidates = {
            "cbc":    (pulp.PULP_CBC_CMD,                           {}),
            "glpk":   (pulp.GLPK_CMD,                               {}),
            "highs":  (getattr(pulp, "HiGHS_CMD",  None),           {}),
            "gurobi": (getattr(pulp, "GUROBI_CMD", None),           {}),
            "scip":   (getattr(pulp, "SCIP_CMD",   None),           {}),
            "xpress": (getattr(pulp, "XPRESS",     None),           {}),
        }
        for name, (cls, _) in candidates.items():
            if cls is None:
                _AVAILABLE_SOLVERS[name] = False
            else:
                try:
                    _AVAILABLE_SOLVERS[name] = cls(msg=0).available()
                except Exception:
                    _AVAILABLE_SOLVERS[name] = False
    except ImportError:
        pass

    available = [k for k in _SOLVER_PRIORITY if _AVAILABLE_SOLVERS.get(k)]
    if available:
        print(f"[solver] Available ILP solvers: {available}  (auto → {available[0]})")
    else:
        print("[solver] WARNING: No ILP solver found — ILP backend will fall back to approx")


_check_solvers()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    cover: Set                    # set of nodes in the MVC
    size: int                     # |cover|
    runtime_seconds: float
    backend_used: str             # 'ilp', 'approx', or 'trivial'
    is_optimal: bool              # True only for ILP solutions
    lp_bound: Optional[int] = None  # LP relaxation lower bound (ILP only)

    @property
    def approximation_ratio(self) -> Optional[float]:
        if self.lp_bound and self.lp_bound > 0:
            return self.size / self.lp_bound
        return None


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class MVCSolver:
    """
    MVC solver wrapper.

    Parameters
    ----------
    backend : str
        'ilp' (default) or 'approx'.
    timeout : int
        Time limit in seconds for ILP solver (default 120).
    fallback_to_approx : bool
        If True, fall back to approximation when ILP times out or fails.
    ilp_threads : int
        Solver threads (default 1).  Ignored by solvers that don't support it.
    ilp_solver : str
        Which ILP solver to use: 'auto' (default — picks fastest available),
        'cbc', 'glpk', 'highs', 'gurobi', or 'scip'.
    """

    KNOWN_SOLVERS = ("auto", "cbc", "glpk", "highs", "gurobi", "scip", "xpress")

    def __init__(
        self,
        backend: str = "ilp",
        timeout: int = 120,
        fallback_to_approx: bool = True,
        ilp_threads: int = 1,
        ilp_solver: str = "auto",
    ):
        assert backend in ("ilp", "approx"), f"Unknown backend: {backend}"
        assert ilp_solver in self.KNOWN_SOLVERS, f"Unknown ILP solver: {ilp_solver}"
        self.backend = backend
        self.timeout = timeout
        self.fallback_to_approx = fallback_to_approx
        self.ilp_threads = ilp_threads
        self.ilp_solver = ilp_solver

    def solve(self, G: nx.Graph) -> SolveResult:
        """Find a Minimum Vertex Cover of G."""
        if G.number_of_nodes() == 0:
            return SolveResult(cover=set(), size=0, runtime_seconds=0.0,
                               backend_used="trivial", is_optimal=True)

        # Isolated nodes never need to be in the cover
        G_work = G.copy()
        isolated = list(nx.isolates(G_work))
        G_work.remove_nodes_from(isolated)

        if G_work.number_of_edges() == 0:
            return SolveResult(cover=set(), size=0, runtime_seconds=0.0,
                               backend_used="trivial", is_optimal=True)

        t0 = time.perf_counter()

        if self.backend == "ilp":
            try:
                result = self._solve_ilp(G_work)
                result.runtime_seconds = time.perf_counter() - t0
                # Sanity check: empty cover on a graph with edges is a bug
                if len(result.cover) == 0 and G_work.number_of_edges() > 0:
                    raise RuntimeError(
                        f"ILP returned empty cover on a graph with "
                        f"{G_work.number_of_edges()} edges — solver malfunction"
                    )
                return result
            except Exception as e:
                msg = f"[solver] ILP failed: {e}"
                if self.fallback_to_approx:
                    print(f"{msg}  →  falling back to 2-approx")
                else:
                    print(msg)
                    raise

        result = self._solve_approx(G_work)
        result.runtime_seconds = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------
    # ILP backend
    # ------------------------------------------------------------------

    def _resolve_solver(self) -> str:
        """Return the concrete solver name to use (never 'auto')."""
        if self.ilp_solver != "auto":
            return self.ilp_solver
        for name in _SOLVER_PRIORITY:
            if _AVAILABLE_SOLVERS.get(name):
                return name
        raise RuntimeError(
            "No ILP solver is available.  Install one of: "
            "pulp (CBC/GLPK bundled), highspy (HiGHS), gurobipy (Gurobi), "
            "pyscipopt (SCIP)."
        )

    def _get_solver_cmd(self, pulp, timeout: int):
        """Return the PuLP solver command object for the resolved solver."""
        name = self._resolve_solver()
        t = {"timeLimit": timeout}
        if name == "cbc":
            return pulp.PULP_CBC_CMD(msg=0, threads=self.ilp_threads, **t)
        if name == "glpk":
            return pulp.GLPK_CMD(msg=0, **t)
        if name == "highs":
            try:
                return pulp.HiGHS_CMD(msg=0, threads=self.ilp_threads, **t)
            except TypeError:
                return pulp.HiGHS_CMD(msg=0, **t)
        if name == "gurobi":
            return pulp.GUROBI_CMD(msg=0, threads=self.ilp_threads, **t)
        if name == "scip":
            return pulp.SCIP_CMD(msg=0, **t)
        if name == "xpress":
            return pulp.XPRESS(msg=0, **t)
        raise ValueError(f"Unknown solver: {name}")

    def _solve_ilp(self, G: nx.Graph) -> SolveResult:
        try:
            import pulp
        except ImportError:
            raise ImportError("PuLP is required.  Install with: pip install pulp")

        solver_name = self._resolve_solver()
        nodes = list(G.nodes())

        # ---- LP relaxation ----
        lp_bound = self._compute_lp_bound(G, nodes, pulp, self._get_solver_cmd(pulp, 10))

        # ---- Integer program ----
        prob = pulp.LpProblem("MVC", pulp.LpMinimize)
        x = {v: pulp.LpVariable(f"x_{i}", cat="Binary") for i, v in enumerate(nodes)}
        prob += pulp.lpSum(x[v] for v in nodes)
        for u, v in G.edges():
            prob += x[u] + x[v] >= 1

        status = prob.solve(self._get_solver_cmd(pulp, self.timeout))

        # PuLP status: 1=Optimal, 0=NotSolved, -1=Infeasible, -2=Unbounded, -3=Undefined
        if status not in (1,):
            raise RuntimeError(
                f"ILP solver ({solver_name}) returned non-optimal status {status} "
                f"({pulp.LpStatus.get(status, '?')})"
            )

        cover = {v for v in nodes if (pulp.value(x[v]) or 0.0) >= 0.5}

        return SolveResult(
            cover=cover,
            size=len(cover),
            runtime_seconds=0.0,
            backend_used=f"ilp:{solver_name}",
            is_optimal=True,
            lp_bound=lp_bound,
        )

    @staticmethod
    def _compute_lp_bound(G: nx.Graph, nodes: list, pulp, solver_cmd) -> Optional[int]:
        """LP relaxation lower bound — uses the same solver as the MIP."""
        try:
            lp_prob = pulp.LpProblem("MVC_LP", pulp.LpMinimize)
            y = {v: pulp.LpVariable(f"y_{i}", lowBound=0.0, upBound=1.0, cat="Continuous")
                 for i, v in enumerate(nodes)}
            lp_prob += pulp.lpSum(y[v] for v in nodes)
            for u, v in G.edges():
                lp_prob += y[u] + y[v] >= 1
            lp_prob.solve(solver_cmd)
            if lp_prob.status == 1:
                return int(np.ceil(pulp.value(lp_prob.objective) - 1e-6))
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Approximation backend
    # ------------------------------------------------------------------

    def _solve_approx(self, G: nx.Graph) -> SolveResult:
        """
        2-approximation MVC:
          - If bipartite: König's theorem (optimal for bipartite graphs)
          - Otherwise: maximal matching → include both endpoints
        """
        if nx.is_bipartite(G):
            cover = self._konig_mvc(G)
        else:
            cover = self._matching_approx(G)

        return SolveResult(
            cover=cover,
            size=len(cover),
            runtime_seconds=0.0,
            backend_used="approx",
            is_optimal=False,
        )

    @staticmethod
    def _matching_approx(G: nx.Graph) -> Set:
        """Standard 2-approximation: pick maximal matching, include both endpoints."""
        cover: Set = set()
        matched: Set = set()
        for u, v in G.edges():
            if u not in matched and v not in matched:
                cover.add(u)
                cover.add(v)
                matched.add(u)
                matched.add(v)
        return cover

    @staticmethod
    def _konig_mvc(G: nx.Graph) -> Set:
        """König's theorem MVC for bipartite graphs (optimal).

        Uses nx.bipartite.color which works on disconnected graphs too.
        """
        coloring = nx.bipartite.color(G)
        top    = {n for n, c in coloring.items() if c == 0}
        bottom = {n for n, c in coloring.items() if c == 1}

        if not top or not bottom:
            return MVCSolver._matching_approx(G)

        # Maximum matching (dict: node -> matched_node, for both sides)
        matching = nx.bipartite.maximum_matching(G, top_nodes=top)

        # Alternating-path BFS from unmatched top nodes
        unmatched_top = top - {u for u in top if u in matching}
        reachable: Set = set(unmatched_top)
        queue = list(unmatched_top)
        while queue:
            u = queue.pop()
            if u in top:
                for v in G.neighbors(u):
                    if v not in reachable and matching.get(u) != v:
                        reachable.add(v)
                        queue.append(v)
            else:
                v = matching.get(u)
                if v is not None and v not in reachable:
                    reachable.add(v)
                    queue.append(v)

        # König: cover = (top \ reachable) ∪ (bottom ∩ reachable)
        return (top - reachable) | (bottom & reachable)


# ---------------------------------------------------------------------------
# Batch solver
# ---------------------------------------------------------------------------

def solve_batch(
    graphs: list,
    backend: str = "ilp",
    timeout: int = 60,
    fallback_to_approx: bool = True,
    ilp_solver: str = "auto",
) -> list:
    """Solve MVC for a list of graphs. Returns list of SolveResult."""
    solver = MVCSolver(backend=backend, timeout=timeout,
                       fallback_to_approx=fallback_to_approx,
                       ilp_solver=ilp_solver)
    return [solver.solve(G) for G in graphs]
