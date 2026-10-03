import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
from train import main
from evaluate import evaluate
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/paper.yaml');a=p.parse_args(); variants=['LL','SS','UO','MN','UA','RB']; out={}
# LL/SS/UO/MN are data-pipeline ablations in the manuscript. Without source-level modality labels, this repository records them as hooks; UA/RB are executable model ablations.
for v in variants:
 if v in {'UA','RB'}:
  ck=main(a.config,disable=[v],tag=f'ablate_{v}');out[v]=evaluate(a.config,ck)
 else: out[v]={'status':'requires source-level modality-tagged KMR data; hook intentionally not fabricated'}
Path('results/ablation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
