import argparse,json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from kmr.utils.config import load_config,seed_everything,device_from_config
from kmr.data.loader import load_bundle,user_level_split,InteractionDataset
from kmr.models.kmr import KMR
from kmr.models.srn import srn_filter

def main(config,disable=None,seed=None,tag=None):
 c=load_config(config); seed=c['seed'] if seed is None else seed; seed_everything(seed); dev=device_from_config(c['training']['device']); b=load_bundle(c['data']['dir']); tr,va,te=user_level_split(b['interactions'],c['data']['train_ratio'],c['data']['val_ratio'],seed)
 mc=dict(c['model']); model=KMR(b['n_users'],b['n_items'],b['n_entities'],b['n_relations'],b['user_features'].shape[1],b['item_features'].shape[1],mc).to(dev); edges=srn_filter(b['edges'],b['n_entities'],mc['srn_threshold']).to(dev); uf=b['user_features'].to(dev); itf=b['item_features'].to(dev); ue=b['user_entity'].to(dev); ie=b['item_entity'].to(dev)
 opt=torch.optim.SGD(model.parameters(),lr=c['training']['learning_rate']); loader=DataLoader(InteractionDataset(tr),batch_size=c['training']['batch_size'],shuffle=True); best=1e99; wait=0; out=Path(c['output_dir'] if tag is None else f"{c['output_dir']}_{tag}"); out.mkdir(parents=True,exist_ok=True)
 for ep in range(c['training']['epochs']):
  model.train(); total=0
  for u,i,y in loader:
   u,i,y=u.to(dev),i.to(dev),y.to(dev); opt.zero_grad(); p,zu,zi,a=model(u,i,uf,itf,edges,ue,ie,disable); loss,_=model.objective(p,y,zu,zi,a); loss.backward(); opt.step(); total+=loss.item()*len(y)
  model.eval(); vals=[]
  with torch.no_grad():
   for u,i,y in DataLoader(InteractionDataset(va),batch_size=4096):
    p,*_=model(u.to(dev),i.to(dev),uf,itf,edges,ue,ie,disable); vals.append(torch.mean((p-y.to(dev))**2).item())
  vl=sum(vals)/max(1,len(vals)); print(f'epoch={ep+1:03d} train_loss={total/max(1,len(tr)):.6f} val_mse={vl:.6f}')
  if vl<best: best=vl; wait=0; torch.save({'state_dict':model.state_dict(),'config':c,'seed':seed,'disable':disable or []},out/'best.pt')
  else:
   wait+=1
   if wait>=c['training'].get('patience',8): break
 return out/'best.pt'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/paper.yaml');p.add_argument('--disable',nargs='*',default=[]);p.add_argument('--seed',type=int);a=p.parse_args();main(a.config,a.disable,a.seed)
