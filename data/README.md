# Dataset Documentation

All datasets are stored in the same JSON format and are interchangeable
through the unified `DatasetLoader` interface (`data/unified.py`).

## JSON format

Every dataset file is a JSON array of records:
```json
[
  {
    "n_nodes": 20,
    "edges":   [[0,1],[1,2],...],
    "mvc":     [1,3,...],
    "source":  "descriptive string"
  },
  ...
]
```

- `n_nodes`: integer, number of nodes (0-indexed: 0 … n_nodes-1)
- `edges`: list of `[u, v]` pairs, undirected, 0-indexed
- `mvc`: list of node ids in the optimal (or near-optimal) MVC, 0-indexed
- `source`: free-form provenance string

---

## erdos — `data/erdos/`

**Source**: PKU-ML/Erdos on HuggingFace  
**Task**: `min_vertex_cover`  
**Size**: 2000 train + 100 test graphs, 4–34 nodes each  
**Labels**: optimal MVC from the original dataset

**Download**:
```bash
python download_dataset.py --all-splits
```

**Citation**:
```
@misc{pku-ml-erdos,
  title  = {PKU-ML/Erdos graph dataset},
  author = {PKU Machine Learning Group},
  url    = {https://huggingface.co/datasets/PKU-ML/Erdos},
}
```

---

## synthetic — `data/synthetic/`

Randomly generated graphs with ILP-computed **optimal** MVC labels.

| Split  | Node range | Graphs/family | Output file               |
|--------|-----------|--------------|---------------------------|
| small  | 20–50     | 200          | `data/synthetic/small.json`  |
| medium | 50–150    | 100          | `data/synthetic/medium.json` |
| large  | 150–500   | 50           | `data/synthetic/large.json`  |

Each split covers four graph families: `erdos_renyi`, `barabasi_albert`,
`watts_strogatz`, `random_regular`.

**Generate**:
```bash
python generate_synthetic.py --split small
python generate_synthetic.py --split medium
python generate_synthetic.py --split large
python generate_synthetic.py --split all          # all three splits
python generate_synthetic.py --split small --n-per-family 50   # faster
```

Graphs where ILP times out are skipped; only optimally-labelled graphs
are saved.

---

## pace — `data/pace/`

**Source**: PACE 2019 Vertex Cover track (exact track, public instances)  
**Labels**: optimal MVC from the official `.sol` files  
**Node range**: varies; typically 10–10 000 nodes

**Download** (manual):
```bash
wget https://pacechallenge.org/files/pace2019-vc-exact-public.tar.gz
tar -xzf pace2019-vc-exact-public.tar.gz -C data/pace/
```

After extraction `data/pace/` should contain `.gr` and `.sol` file pairs.
Graphs without a matching `.sol` file are skipped.

**Load in Python**:
```python
from data.pace import load_pace
graphs = load_pace(min_nodes=10, max_nodes=500)
```

**Citation**:
```
@inproceedings{pace2019,
  title     = {PACE 2019 Parameterized Algorithms and Computational Experiments Challenge},
  booktitle = {14th International Symposium on Parameterized and Exact Computation (IPEC 2019)},
  year      = {2019},
  url       = {https://pacechallenge.org/2019/},
}
```

---

## tudataset — `data/tudatasets/`

**Source**: TUDataset benchmark collection (Morris et al. 2020)  
**Labels**: ILP-computed MVC (not from original datasets; original labels
are discarded)  
**Cached**: `data/tudatasets/{NAME}_mvc.json`

Supported datasets:

| Name          | Graphs | Avg nodes | Domain              |
|---------------|--------|-----------|---------------------|
| MUTAG         | 188    | 17.9      | Chemical (mutagenicity) |
| NCI1          | 4110   | 29.9      | Chemical (NCI)      |
| NCI109        | 4127   | 29.7      | Chemical (NCI)      |
| PROTEINS      | 1113   | 39.1      | Protein structure   |
| DD            | 1178   | 284.3     | Protein (D&D)       |
| IMDB-BINARY   | 1000   | 19.8      | Social (movie)      |
| REDDIT-BINARY | 2000   | 429.6     | Social (Reddit)     |

These datasets are structurally interesting in the context of the symmetry
kernel literature (Kuhar & Cibej, 2026).

**Download and compute labels**:
```bash
python compute_tudataset_mvc.py --dataset MUTAG
python compute_tudataset_mvc.py --dataset NCI1 --max-graphs 500
python compute_tudataset_mvc.py --dataset PROTEINS --solver-timeout 120
```

Raw data is downloaded automatically from
`https://www.chrsmrrs.com/graphkerneldatasets/{NAME}.zip`.

**Citation**:
```
@article{morris2020tudataset,
  title   = {TUDataset: A collection of benchmark datasets for learning with graphs},
  author  = {Morris, Christopher and others},
  journal = {arXiv:2007.08663},
  year    = {2020},
}
```

---

## Using multiple datasets together

```python
from data.unified import DatasetLoader

loader = DatasetLoader()
train, test = loader.load(
    sources=["erdos", "synthetic:small", "tudataset:MUTAG"],
    min_nodes=5,
    max_nodes=200,
    train_ratio=0.8,
    seed=42,
)
```

Or from the command line:
```bash
python train.py --datasets erdos,synthetic:small
python train.py --datasets erdos,synthetic:small,synthetic:medium,pace
python train.py --datasets erdos,tudataset:MUTAG,tudataset:NCI1
```
