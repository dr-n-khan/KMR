import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from kmr.utils.config import load_config
from kmr.data.loader import load_bundle
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/paper.yaml');p.add_argument('--folds',type=int,default=5);a=p.parse_args();c=load_config(a.config);b=load_bundle(c['data']['dir']); users=np.array(sorted(b['interactions'].user_idx.unique()));kf=KFold(a.folds,shuffle=True,random_state=c['seed']);splits=[]
for n,(tr,te) in enumerate(kf.split(users),1):splits.append({'fold':n,'train_users':users[tr].tolist(),'test_users':users[te].tolist()})
Path('results/cv_splits.json').write_text(json.dumps(splits,indent=2));print('Saved leakage-free user-level folds. Use these split IDs in paper-scale training runs.')
