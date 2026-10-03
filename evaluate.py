import argparse,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from kmr.utils.config import load_config,device_from_config
from kmr.data.loader import load_bundle,user_level_split,InteractionDataset
from kmr.models.kmr import KMR
from kmr.models.srn import srn_filter
from kmr.evaluation.metrics import ranking_by_user,errors,coverage,novelty,diversity

def evaluate(config,checkpoint):
 c=load_config(config); ck=torch.load(checkpoint,map_location='cpu',weights_only=False); seed=ck.get('seed',c['seed']); b=load_bundle(c['data']['dir']); _,_,te=user_level_split(b['interactions'],c['data']['train_ratio'],c['data']['val_ratio'],seed); dev=device_from_config(c['training']['device']); m=KMR(b['n_users'],b['n_items'],b['n_entities'],b['n_relations'],b['user_features'].shape[1],b['item_features'].shape[1],c['model']).to(dev);m.load_state_dict(ck['state_dict']);m.eval(); edges=srn_filter(b['edges'],b['n_entities'],c['model']['srn_threshold']).to(dev);uf=b['user_features'].to(dev);itf=b['item_features'].to(dev);ue=b['user_entity'].to(dev);ie=b['item_entity'].to(dev); scores=[]
 with torch.no_grad():
  for u,i,y in DataLoader(InteractionDataset(te),batch_size=4096): scores.extend(m(u.to(dev),i.to(dev),uf,itf,edges,ue,ie,ck.get('disable',[]))[0].cpu().numpy())
 scores=np.asarray(scores); metrics={**errors(te.label.values,scores),**ranking_by_user(te,scores,c['evaluation']['ks'])}; tf=te.copy();tf['score']=scores; k=5;recs=[]
 for _,g in tf.groupby('user_idx'): recs.append(g.sort_values('score',ascending=False).item_idx.head(k).astype(int).tolist())
 pop=te.item_idx.value_counts().to_dict(); metrics.update({'Coverage':coverage(recs,b['n_items']),'Novelty':novelty(recs,pop,len(te)),'Diversity':diversity(recs,b['item_features'].numpy())}); print(json.dumps(metrics,indent=2)); Path(checkpoint).with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)); return metrics
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/paper.yaml');p.add_argument('--checkpoint',required=True);a=p.parse_args();evaluate(a.config,a.checkpoint)
