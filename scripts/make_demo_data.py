import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
from kmr.data.synthetic import make_demo
p=argparse.ArgumentParser();p.add_argument('--out',default='data/processed/demo');p.add_argument('--users',type=int,default=300);p.add_argument('--items',type=int,default=120);p.add_argument('--seed',type=int,default=42);a=p.parse_args();make_demo(a.out,a.users,a.items,a.seed);print('Wrote demo data to',a.out)
