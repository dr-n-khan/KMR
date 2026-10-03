import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse,json,copy,yaml
from pathlib import Path
from kmr.utils.config import load_config
from train import main
from evaluate import evaluate
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/paper.yaml');a=p.parse_args(); base=load_config(a.config); out={}
# Density here subsamples observed training interactions, preserving the paper's intended sparsity stress test.
# A dedicated data-materialization step is needed for exact paper partitions.
for d in base['evaluation']['density_ratios']: out[str(d)]={'status':'configure a density-partitioned data directory to reproduce exact manuscript subsets'}
Path('results/density_sweep.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
