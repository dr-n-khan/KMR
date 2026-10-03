import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
from train import main
from evaluate import evaluate
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/paper.yaml');p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);a=p.parse_args(); rows=[]
for s in a.seeds:
 ck=main(a.config,seed=s,tag=f'seed{s}'); rows.append({'seed':s,**evaluate(a.config,ck)})
Path('results/repeated.json').write_text(json.dumps(rows,indent=2))
