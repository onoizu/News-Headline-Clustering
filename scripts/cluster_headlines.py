"""Reproducible headline clustering; models are loaded only when requested."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd


def clean_text(value: str) -> str:
    """Match the original regex tokenizer while leaving raw text available."""
    return " ".join(t for t in re.findall(r"[a-zA-Z']+", value.lower())
                    if len(t) > 1 and any(c.isalpha() for c in t))


def prepare_data(frame: pd.DataFrame, sample_size: int, seed: int):
    """Deduplicate first, then sample without replacement by publication year."""
    required = {"publish_date", "headline_text"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    data = frame.copy()
    data["source_row"] = np.arange(len(data))
    audit = {"input_rows": len(data)}
    data = data.dropna(subset=["headline_text", "publish_date"]).copy()
    data["headline_text"] = data["headline_text"].astype(str).str.strip()
    data["clean_text"] = data["headline_text"].map(clean_text)
    dates = pd.to_datetime(data["publish_date"].astype(str), format="%Y%m%d",
                           errors="coerce")
    data["year"] = dates.dt.year
    data = data.loc[dates.notna() & data["clean_text"].ne("")].copy()
    audit["invalid_or_empty_rows"] = audit["input_rows"] - len(data)
    before = len(data)
    data = data.drop_duplicates("headline_text", keep="first")
    audit["duplicate_headlines_removed"] = before - len(data)
    audit["eligible_rows"] = len(data)
    if len(data) < 2:
        raise ValueError("At least two valid, unique headlines are required")
    counts = data.groupby("year", sort=True).size()
    n = min(sample_size, len(data))
    # Largest-remainder allocation gives exactly n rows, without resetting
    # indices before excluding already selected rows.
    quotas = counts * n / len(data)
    allocation = np.floor(quotas).astype(int)
    remainder = n - int(allocation.sum())
    order = (quotas - allocation).sort_values(ascending=False, kind="stable")
    allocation.loc[order.index[:remainder]] += 1
    parts = [data.loc[data["year"].eq(year)].sample(n=int(k), random_state=seed)
             for year, k in allocation.items() if k]
    result = pd.concat(parts).sort_values(["publish_date", "source_row"])
    result["year"] = result["year"].astype(int)
    result = result.reset_index(drop=True)
    audit["sampled_rows"] = len(result)
    audit["sampled_year_counts"] = {
        str(k): int(v) for k, v in result.groupby("year").size().items()}
    return result, audit


def make_features(data, args):
    if args.embedding == "sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model)
        # Sentence encoders receive original headlines, not stopword-stripped text.
        return model.encode(data["headline_text"].tolist(),
                            batch_size=args.batch_size, normalize_embeddings=True,
                            show_progress_bar=True), {"model": args.model}
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000,
                                 token_pattern=r"(?u)\b[a-zA-Z'][a-zA-Z']+\b")
    matrix = vectorizer.fit_transform(data["clean_text"])
    dimensions = min(args.lsa_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
    if dimensions < 1:
        raise ValueError("Too little vocabulary for TF-IDF/LSA")
    svd = TruncatedSVD(n_components=dimensions, random_state=args.seed)
    features = normalize(svd.fit_transform(matrix))
    return features, {"vocabulary_size": len(vectorizer.vocabulary_),
                      "lsa_dimensions": dimensions,
                      "explained_variance_ratio": float(svd.explained_variance_ratio_.sum())}


def reduce_features(features, args):
    if args.reducer == "none":
        return features
    if len(features) < 5:
        raise ValueError("UMAP needs more data; use --reducer none for tiny examples")
    import umap
    return umap.UMAP(n_components=min(15, len(features) - 2),
                     n_neighbors=min(15, len(features) - 1), min_dist=0.0,
                     metric="cosine", random_state=args.seed,
                     init="random").fit_transform(features)


def cluster(features, args):
    if args.clusterer == "hdbscan":
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                               min_samples=args.min_samples,
                               cluster_selection_epsilon=args.cluster_epsilon,
                               metric="euclidean").fit_predict(features)
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=args.eps, min_samples=args.min_samples,
                  metric="euclidean").fit_predict(features)


def evaluate(features, labels, seed):
    from sklearn.metrics import silhouette_score
    keep = labels != -1
    counts = Counter(int(x) for x in labels if x != -1)
    result = {"n_clusters": len(counts), "noise_count": int((~keep).sum()),
              "noise_ratio": float((~keep).mean()),
              "cluster_sizes": dict(sorted(counts.items())),
              "silhouette": None, "silhouette_samples": 0,
              "silhouette_space": "clustering feature space, non-noise points only"}
    if 1 < len(counts) < int(keep.sum()):
        # Bound the quadratic pairwise-distance work, preserving natural noise.
        indices = np.flatnonzero(keep)
        rng = np.random.default_rng(seed)
        if len(indices) > 2000:
            indices = np.sort(rng.choice(indices, 2000, replace=False))
        subset = labels[indices]
        if 1 < len(np.unique(subset)) < len(subset):
            score = float(silhouette_score(features[indices], subset))
            result["silhouette"] = score if math.isfinite(score) else None
            result["silhouette_samples"] = len(indices)
    return result


def summarize_topics(data, labels):
    """Use one shared TF-IDF vocabulary to summarize non-noise clusters."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    rows = []
    if not np.any(labels != -1):
        return rows
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    try:
        matrix = vectorizer.fit_transform(data["clean_text"])
    except ValueError:
        return rows
    terms = vectorizer.get_feature_names_out()
    for label in sorted(set(labels) - {-1}):
        mask = labels == label
        weights = np.asarray(matrix[mask].mean(axis=0)).ravel()
        top = [terms[i] for i in np.argsort(-weights, kind="stable")[:8]
               if weights[i] > 0]
        examples = data.loc[mask, "headline_text"].head(3).tolist()
        rows.append({"cluster": int(label), "size": int(mask.sum()),
                     "keywords": list(top), "example_headlines": examples})
    return rows


def save_plot(features, labels, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2).fit_transform(features) if features.shape[1] > 1 else np.column_stack([features[:, 0], np.zeros(len(features))])
    fig, ax = plt.subplots(figsize=(9, 6))
    keep = labels != -1
    ax.scatter(xy[~keep, 0], xy[~keep, 1], c="#b7bcc6", s=5, alpha=0.35,
               label="Noise")
    if keep.any():
        ax.scatter(xy[keep, 0], xy[keep, 1], c=labels[keep], cmap="tab20",
                   s=5, alpha=0.65)
    ax.set(title="Headline clusters: PCA display projection",
           xlabel="Display component 1", ylabel="Display component 2")
    ax.text(0, -0.14, "Display only; clustering and silhouette use the full clustering space.",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "clusters.png", dpi=160)
    plt.close(fig)


def package_versions():
    result = {}
    for name in ["numpy", "pandas", "scikit-learn", "sentence-transformers",
                 "umap-learn", "hdbscan", "matplotlib"]:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding", choices=["sbert", "tfidf"], default="sbert")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lsa-components", type=int, default=100)
    parser.add_argument("--reducer", choices=["umap", "none"], default="umap")
    parser.add_argument("--clusterer", choices=["hdbscan", "dbscan"], default="hdbscan")
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--cluster-epsilon", type=float, default=0.12)
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN radius")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.lsa_components < 1 or args.min_samples < 1 or args.min_cluster_size < 2 or args.eps <= 0 or args.cluster_epsilon < 0:
        raise ValueError("Invalid model or clustering parameters")
    # Refuse to overwrite an input CSV through output naming.
    destinations = [args.output / n for n in ["assignments.csv", "topics.json", "metrics.json", "run_info.json", "clusters.png"]]
    if args.input.resolve() in [p.resolve() for p in destinations]:
        raise ValueError("Choose a different output directory: input would be overwritten")
    data, audit = prepare_data(pd.read_csv(args.input, dtype={"publish_date": str}),
                               args.sample_size, args.seed)
    features, embedding_info = make_features(data, args)
    reduced = reduce_features(features, args)
    labels = cluster(reduced, args)
    metrics = evaluate(reduced, labels, args.seed)
    topics = summarize_topics(data, labels)
    args.output.mkdir(parents=True, exist_ok=True)
    data.assign(cluster=labels).to_csv(args.output / "assignments.csv", index=False)
    metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "python": platform.python_version(), "packages": package_versions(),
                "audit": audit, "embedding": embedding_info,
                "embedding_shape": list(features.shape), "clustering_shape": list(reduced.shape)}
    for name, value in [("topics.json", topics), ("metrics.json", metrics), ("run_info.json", metadata)]:
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    if args.plot:
        save_plot(reduced, labels, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
