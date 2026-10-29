# 📰 News Headline Clustering with Sentence-BERT

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Task](https://img.shields.io/badge/Task-Unsupervised%20NLP-6C63FF)
![Method](https://img.shields.io/badge/Pipeline-SBERT%20%E2%86%92%20UMAP%20%E2%86%92%20HDBSCAN-00897B)

**Discovering topics in short news headlines through semantic embeddings and density-based clustering.**

A DSAI4203 course group project at The Hong Kong Polytechnic University, exploring a **50,000-headline sample** from the ABC *A Million News Headlines* dataset. The project investigated lexical and semantic representations, with **Sentence-BERT → UMAP → HDBSCAN** as the final experimental pipeline.

This repository contains the original experiment notebook and a maintained Python implementation with reproducible sampling, structured outputs, and a small synthetic demo.

## 🎯 Project overview

News headlines are short and often use different words to describe related events. This project explores how sentence-level representations can group semantically related headlines without manually assigned topic labels.

- **Data preparation:** remove duplicate headlines, validate dates, clean text, and sample proportionally by publication year.
- **Representation exploration:** TF-IDF/LSA and Sentence-BERT in the included notebook; earlier project work also explored Word2Vec.
- **Semantic clustering:** encode headlines with `all-MiniLM-L6-v2`, reduce their dimensions with UMAP, and identify dense groups with HDBSCAN.
- **Interpretation:** inspect cluster keywords, example headlines, sizes, noise, and a display projection.

### My contribution

**Meihui An · data preparation and feature representation**

My contributions to the course group project included headline deduplication, year-stratified sampling, text cleaning and tokenization, exploratory data analysis, and experiments with text representations. The complete clustering workflow was a **group effort**.

The command-line implementation in this repository was subsequently organized from the course experiments for clearer execution and inspection. It is a refactored implementation, rather than a claim that the original notebook was already a production system.

## 🔎 How it works

```text
CSV headlines + publication dates
              │
     Validate and deduplicate
              │
  Year-stratified sample (up to 50,000)
              │
 Sentence-BERT embeddings (384 dimensions)
              │
      UMAP (15 dimensions)
              │
           HDBSCAN
              │
 Assignments · Keywords · Metrics · Run metadata
```

The maintained pipeline keeps HDBSCAN's `-1` noise labels. It does not force every headline into a topic. A separate **TF-IDF → LSA → DBSCAN** route provides a lightweight baseline and a quick installation check.

## 🚀 Quick start

Use **Python 3.11 or 3.12** as the recommended starting point. Commands below are run from the repository root.

```bash
git clone git@github.com:onoizu/News-Headline-Clustering.git
cd News-Headline-Clustering
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### 1. Try the lightweight demo

```bash
python scripts/cluster_headlines.py \
  --input data/demo_headlines.csv \
  --output outputs/demo \
  --embedding tfidf --reducer none --clusterer dbscan \
  --eps 0.9 --min-samples 2 --plot
```

The included CSV contains **24 synthetic headlines** about football, astronomy, and banking. It checks that the workflow runs; it is not a subset of the original dataset or evidence of real-world clustering quality.

### 2. Run the semantic pipeline

Install the additional embedding and clustering dependencies:

```bash
python -m pip install -r requirements-semantic.txt
```

Download [A Million News Headlines from Kaggle](https://www.kaggle.com/datasets/therohk/million-headlines) and place `abcnews-date-text.csv` in `data/`. The full dataset is not redistributed here; follow its source terms.

Expected input columns:

```csv
publish_date,headline_text
20200101,example news headline
20200102,another example news headline
```

Dates must use `YYYYMMDD`. Additional columns are allowed. Then run:

```bash
python scripts/cluster_headlines.py \
  --input data/abcnews-date-text.csv \
  --output outputs/sbert \
  --sample-size 50000 --seed 42 --plot
```

The first semantic run downloads the pretrained Sentence-BERT model and requires internet access. No external API key is needed. CPU execution is supported; embedding 50,000 headlines and fitting UMAP can take time and memory. Start with `--sample-size 2000 --min-cluster-size 20` to check the setup before a larger run; those settings define a different experiment.

### 3. Run the tests

```bash
python -m unittest discover -s tests -v
```

Tests cover deterministic sampling, exact sample size and source-row uniqueness, invalid inputs, undefined silhouette cases, and end-to-end output consistency using the synthetic demo.

## 📁 Repository structure

```text
News-Headline-Clustering/
├── README.md
├── requirements.txt                 # Lightweight baseline and tests
├── requirements-semantic.txt        # SBERT, UMAP, HDBSCAN
├── requirements-notebooks.txt       # Historical notebook dependencies
├── data/
│   └── demo_headlines.csv           # Synthetic smoke-test input
├── scripts/
│   └── cluster_headlines.py         # Maintained command-line pipeline
├── notebooks/
│   └── 01_original_experiments.ipynb
├── docs/
│   ├── methodology.md              # Experimental choices and limitations
│   └── validation.md               # What was actually checked
└── tests/
    └── test_pipeline.py
```

Generated runs go into `outputs/`, which is ignored by Git.

## 📦 Outputs

Each run writes:

- **`assignments.csv`** — original headline, cleaned text, publication date, year, zero-based source row, and cluster label. `-1` means noise.
- **`topics.json`** — each non-noise cluster's size, up to eight mean TF-IDF keywords, and three example headlines. Keywords are descriptive cues, not verified topic names.
- **`metrics.json`** — cluster count, cluster sizes, noise count/ratio, and silhouette score where defined.
- **`run_info.json`** — command arguments, input SHA-256, Python and package versions, sampling audit, and feature dimensions.
- **`clusters.png`** — optional PCA display projection when `--plot` is passed. Clustering and metrics use the full clustering feature space, not these two display coordinates.

Silhouette is computed on non-noise points in the clustering feature space, with a deterministic sample of at most 2,000 points to limit pairwise-distance work. It is `null` when the available labels do not support a valid calculation. Compare scores only when representation, distance metric, noise treatment, and evaluation protocol are comparable.

## ⚙️ Main settings

Defaults reproduce the **main method choices**, rather than guaranteeing identical historical labels:

- `--embedding sbert`: normalized `sentence-transformers/all-MiniLM-L6-v2` embeddings of original headlines.
- `--reducer umap`: cosine input distance, 15 dimensions, 15 neighbors, `min_dist=0.0`, random initialization, seed 42. Dimensions/neighbors are reduced for small inputs.
- `--clusterer hdbscan`: Euclidean distance, `min_cluster_size=100`, `min_samples=2`, `cluster_selection_epsilon=0.12`.
- `--sample-size 50000`: proportional year quotas with largest-remainder allocation, sampled without replacement. Uses all eligible rows if fewer are available.
- `--embedding tfidf`: up to 5,000 English-stopword-filtered terms, followed by LSA and L2 normalization. `--lsa-components` defaults to 100 and adapts to small inputs.
- `--clusterer dbscan`: `--eps` defaults to 0.5; tune this for the chosen feature space.

Run `python scripts/cluster_headlines.py --help` for every option. Seeds improve repeatability, but model revisions, library versions, hardware, and input dataset editions can still affect results.

## 🧪 Experiments and interpretation

The original course experiments used 50,000 headlines and compared feature representations before exploring semantic density clustering. The archived notebook preserves that development process, including EDA, TF-IDF/LSA, SBERT, and experimental post-processing.

The maintained script makes several deliberate changes: robust sampling without duplicate source rows, original-text input to the sentence encoder, explicit noise retention, bounded silhouette evaluation, and versioned run metadata. Therefore its outputs should not be presented as an exact replay of every notebook cell.

Historical notebook outputs were not fully consistent across cells, so **no single historical silhouette score is advertised as a verified benchmark**. The original full CSV was unavailable during this repository preparation; the 50,000-headline SBERT experiment has not been rerun here. See [validation](docs/validation.md) for the checks actually completed and [methodology](docs/methodology.md) for details of the exploratory weighting and noise-reassignment steps.

## 💡 Limitations and next steps

- Short headlines can be ambiguous, and a density cluster need not correspond to one coherent editorial topic.
- UMAP and clustering hyperparameters can change cluster granularity and the noise fraction.
- Silhouette and visual separation do not establish semantic correctness; representative-headline review and human topic assessment are useful complements.
- The workflow clusters a batch; it does not implement production ingestion or assignment of future headlines.
- Further work could compare multiple seeds, review samples from every cluster, assess stability, and cache embeddings for faster experiments.

## 🙏 Acknowledgements

Dataset: [A Million News Headlines](https://www.kaggle.com/datasets/therohk/million-headlines), containing ABC news headlines. Libraries used include Sentence Transformers, UMAP, HDBSCAN, scikit-learn, NumPy, pandas, and Matplotlib.

This repository presents an academic group project and the maintainer's stated contributions. Dataset and pretrained-model terms remain with their respective providers; no new reuse license for the original team code is asserted here.
