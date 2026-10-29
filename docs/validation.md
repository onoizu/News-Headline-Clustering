# Validation record

Repository preparation date: **2026-10-30**. Validation environment: macOS ARM64, Python **3.12.14**, in an isolated virtual environment.

## Checks completed

- Installed `requirements.txt`; `python -m pip check` reported no broken requirements.
- Compiled the archived notebook's Python code cells to check syntax. This is not a notebook execution check.
- Ran `python -m unittest discover -s tests -v`: four tests covering sampling, invalid input, undefined silhouette cases, and the complete baseline CLI with plots.
- Ran the documented synthetic-demo command independently. It produced all five output files, including a PNG plot.

Installed numerical/plotting versions: NumPy **2.5.4**, pandas **2.3.3**, scikit-learn **1.9.1**, SciPy **1.18.1**, Matplotlib **3.11.2**. The dependency files specify allowed ranges; these are the versions observed during this check.

## Synthetic demo result

For the 24-row included input, TF-IDF/LSA + DBSCAN with `eps=0.9` and `min_samples=2` produced four clusters, one noise point, and a non-noise silhouette of approximately **0.3066** across 23 points. The planted text themes do not imply that the algorithm must recover exactly three clusters: one theme splits under these settings. This result is a software smoke test, not a benchmark or a measure of accuracy on ABC headlines.

## Not validated in this preparation

The original ABC CSV was not present in the supplied project directory. The 50,000-headline SBERT → UMAP → HDBSCAN experiment, semantic dependency installation, pretrained-model download, and full historical notebook execution were not rerun. No historical metric is represented as independently reproduced. The repository provides those implementation paths and describes their limitations so a future run can produce a traceable result.
