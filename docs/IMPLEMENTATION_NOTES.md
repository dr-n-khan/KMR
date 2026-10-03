# Manuscript → code mapping

| Manuscript component | Code |
|---|---|
| User opinion adjustment, Eq. 2 and preference alignment | kmr/models/kmr.py |
| SRN, Eqs. 5–11 / Algorithm 1 | kmr/models/srn.py |
| Patient representation binding, Eqs. 12–18 | KMR.representations() |
| MDP representation binding, Eqs. 19–22 | KMR.representations() |
| Bilinear DMF recommendation, Eqs. 23–25 | KMR.forward() |
| Alignment/sparsity losses, Eqs. 26–27 | kmr/models/losses.py |
| Threshold-sensitive loss, Eq. 28 | threshold_sensitive_loss() |
| Composite objective, Eqs. 29–35 | KMR.objective() + train.py |
| NDCG/RMSE/MAE/Coverage/Novelty/Diversity | kmr/evaluation/metrics.py |
| Paired t-test, Wilcoxon, 95% CI, paired Cohen d | kmr/evaluation/statistics.py |

## Important assumptions / unavailable details

1. The manuscript describes LLDA/BERT preprocessing but does not identify an exact BERT checkpoint or publish raw text labels. The repository provides the interface and does not claim to reconstruct unavailable source data.
2. The exact feature dimensionalities for patient and MDP attributes are not specified. Numeric columns are discovered from CSVs and projected to the 64-dimensional latent space.
3. SRN's temporal tendency uses recent/past interaction quantities not present in the published dataset schema. The runnable SRN implementation uses neighborhood overlap as a documented proxy. Replace this with timestamp-derived γ/κ when those fields are available.
4. Eq. 27 is implemented literally as an L1 penalty on attention weights. If attention weights are softmax-normalized, their L1 sum is constant; this is retained for paper fidelity rather than silently changing the method.
5. The manuscript's sentiment normalization can be undefined for a negative summed polarity under a literal square root. The core model therefore leaves sentiment as a data-preprocessing hook until the exact intended transformation is specified.
6. The third-party baselines are not reimplemented under the KMR name. Reproduce them from their official implementations/settings and compare their exported predictions with the statistics module.
7. Clinical ontology priors are represented by prior embedding tables initialized to zero. Load mapped ICD-10/DSM-5-derived vectors when licensed mappings are available.

These boundaries make the repository runnable...