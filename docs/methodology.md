# Methodology and experiment notes

## Original project and maintained implementation

The course project investigated unsupervised topic discovery in a 50,000-headline sample. The archived notebook is the final supplied course notebook, with execution counts and saved outputs removed, identifying notebook metadata removed, and its CSV path adjusted to `../data/abcnews-date-text.csv`. Open it from `notebooks/` after installing `requirements-notebooks.txt`. It preserves exploratory code and is not a guaranteed clean sequential execution; use the command-line script as the maintained entry point.

The notebook contains TF-IDF/LSA and SBERT experiments. Word2Vec was explored in earlier project work but is not implemented in the maintained script. No clustering model is trained from scratch in the SBERT branch: a pretrained sentence encoder supplies the representations, followed by unsupervised dimensionality reduction and clustering.

## Sampling

The maintained script validates dates and non-empty cleaned text, removes exact duplicate stripped headline strings, then allocates a target sample proportionally across publication years. Integer quotas use the largest-remainder rule so the total is exact. Sampling is without replacement and retains a source-row identifier. If the target is smaller than the number of years, some years may receive no samples. Duplicates are not removed by normalized semantic meaning; differently cased or worded headlines may remain.

## Representation and clustering

The sentence encoder receives original text to preserve sentence context. The lexical baseline uses lowercased regex tokens, English stopword filtering, TF-IDF, truncated SVD, and normalization. Neither branch uses the publication year as a clustering feature; dates control sampling and support inspection.

The semantic branch reduces normalized embeddings using UMAP with cosine input distance, then fits HDBSCAN using Euclidean distance in the reduced space. Random UMAP initialization is explicit in the maintained code. Cluster count is inferred from the density structure; the algorithm may return noise or even no clusters for unsuitable parameters or small datasets.

## Exploratory TF-IDF scaling

The notebook labels one experiment “TF-IDF weighted SBERT.” Its implementation scales each entire sentence embedding by a positive scalar derived from average TF-IDF, then L2-normalizes it. For a positive scalar `a`, `normalize(a * v) = normalize(v)`. This does not introduce word-level weighting into the sentence embedding and is not evidence of an improved hybrid representation. The maintained pipeline uses ordinary normalized SBERT embeddings.

## Noise reassignment

The historical notebook explores k-nearest-neighbor reassignment of noise labels. Its procedure uses neighbors from the full point set and updates labels iteratively, so decisions can depend on processing order. Reducing the number of noise points alone does not demonstrate better topic quality. The maintained script leaves label `-1` intact and excludes noise from silhouette calculation and topic summaries.

## Evaluation

The maintained silhouette protocol uses Euclidean distances in the actual clustering feature space, excluding noise, with a seeded sample of at most 2,000 points. The sample is uniform over non-noise points, not stratified by cluster. If the sampled labels do not permit a valid silhouette calculation, the score is null. Scores from different representations, projections, or noise treatments are not directly interchangeable.

The PCA chart is a display aid. Its geometry can hide or exaggerate separation, and its coordinates are not used for scoring. Cluster keywords are averages from a shared TF-IDF vocabulary; examples are the first three headlines in sampled date order, not centroid-selected representatives. Review raw headlines before assigning a semantic topic name.

The historical notebook contains inconsistent metrics across cells, and the original full input CSV was not available during repository preparation. Historical figures therefore remain exploratory rather than reproduced benchmark results. Package constraints are compatibility ranges, not an exact lockfile; each completed run records installed versions and input hash for traceability.
