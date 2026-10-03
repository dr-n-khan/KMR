# KMR — Decision Support Recommendations for Managing Alzheimer's Dementia

Reference implementation of the KMR framework described in the accompanying paper. The repository implements the paper's end-to-end computational pipeline: preprocessing hooks, opinion adjustment, relation-aware graph reasoning, semantic-relevance subgraph sampling (SRN), patient/MDP representation binding, DMF-style recommendation, the threshold-sensitive loss, ontology alignment, sparsity/L2 regularization, user-level splits, Top-K evaluation, ablations, repeated runs, 5-fold user-level CV, statistical tests, and inference benchmarking.

> Research code / not medical advice. This software is for reproducibility and research. It must not be used to make clinical decisions without independent clinical validation.

## What is reproducible here

The paper specifies KGE dimension 64, KGE weight 0.001, λ1=0.50, λ2=0.45, λ3=1e-7, learning rate 0.0007, batch size 1024, 50 iterations, dropout search range [0.25, 0.85], K ∈ {1,2,3,5,7,10}, density ∈ {30,40,50,60,70,80}%, SRN threshold 30%, and τ=0.3. These are the defaults in `configs/paper.yaml`.

The original heterogeneous KMR dataset is not bundled because the paper combines public web material with controlled-access clinical repositories and states that the populations are semantically integrated rather than patient-linked. This repository therefore includes (1) a documented CSV interface and (2) a deterministic synthetic-data generator so the complete code path can be tested immediately. Place authorized data under `data/raw/` using the schemas below.

## Quick start

bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/make_demo_data.py --out data/processed/demo --users 300 --items 120
python train.py --config configs/demo.yaml
python evaluate.py --config configs/demo.yaml --checkpoint results/demo/best.pt
pytest -q

For paper-scale settings, use:

bash
python train.py --config configs/paper.yaml
python scripts/run_repeated.py --config configs/paper.yaml
python scripts/run_cv.py --config configs/paper.yaml
python scripts/run_ablation.py --config configs/paper.yaml
python scripts/run_density_sweep.py --config configs/paper.yaml
python scripts/benchmark_inference.py --config configs/paper.yaml

## Expected data

interactions.csv: user_id,item_id,label[,explicit,implicit,review]  
kg.csv: head,relation,tail  
users.csv: user_id,age,sex,stage[,health_text,context_1,...]  
items.csv: item_id,item_type[,dosage,nutrient_1,...]

IDs may be strings. The loader creates contiguous internal indices. The seven relation labels reported in the paper are treats, causes, associated_with, recommended_for, contraindicated_with, belongs_to, and interacts_with; other relations are accepted for extensibility.

## Repository map

- kmr/data/: loading, preprocessing, user-level splitting, synthetic demo data
- kmr/models/: KMR modules, SRN sampling, losses
- kmr/evaluation/: ranking/error/quality metrics and paired statistics
- scripts/: experiments matching the paper's analyses
- configs/: paper and lightweight demo configurations
- tests/: smoke/unit tests
- docs/IMPLEMENTATION_NOTES.md: explicit manuscript-to-code mapping and assumptions

## Reproducibility notes

The manuscript does not provide raw source records, exact BERT checkpoint, all feature dimensions, every selected dropout/weight value, or executable implementations of third-party baselines. The code therefore does not fabricate those missing details. Configurable defaults are marked in configs/*.yaml; baseline result comparison can ingest externally generated predictions/results. The LLDA/BERT stage is represented by reproducible preprocessing/embedding hooks, with an optional Hugging Face encoder if installed and configured.

## Citation

If you use this repository, cite the associated manuscript. Replace the placeholder below with the final bibliographic record/DOI after publication.

bibtex
@article{khan2026decision,
  title={Decision Support Recommendations for Managing Alzheimer's Dementia},
  author={Khan, Nasrullah and Shah, Zubair},
  year={2026},
  journal={Informatics in Medicine Unlocked}
}

## License

MIT for the source code in this repository. Dataset licenses/terms remain those of their original providers and are not changed by this repository.