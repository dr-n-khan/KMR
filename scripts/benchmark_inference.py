import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse,time,torch
from kmr.utils.config import load_config,device_from_config
from kmr.data.loader import load_bundle
from kmr.models.kmr import KMR
from kmr.models.srn import srn_filter
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/paper.yaml');p.add_argument('--checkpoint',required=True);a=p.parse_args();c=load_config(a.config);b=load_bundle(c['data']['dir']);dev=device_from_config(c['training']['device']);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=KMR(b['n_users'],b['n_items'],b['n_entities'],b['n_relations'],b['user_features'].shape[1],b['item_features'].shape[1],c['model']).to(dev);m.load_state_dict(ck['state_dict']);m.eval();E=srn_filter(b['edges'],b['n_entities'],c['model']['srn_threshold']).to(dev);uf=b['user_features'].to(dev);it=b['item_features'].to(dev);ue=b['user_entity'].to(dev);ie=b['item_entity'].to(dev);bs=c['training']['batch_size'];u=torch.arange(bs,device=dev)%b['n_users'];i=torch.arange(bs,device=dev)%b['n_items']
with torch.no_grad():
 t=time.perf_counter()
 while time.perf_counter()-t<3:m(u,i,uf,it,E,ue,ie)
 if dev.type=='cuda':torch.cuda.synchronize()
 ts=[]
 for _ in range(20):
  t=time.perf_counter();m(u,i,uf,it,E,ue,ie)
  if dev.type=='cuda':torch.cuda.synchronize()
  ts.append(time.perf_counter()-t)
print(f'batch={bs} latency_ms={1000*sum(ts)/len(ts):.3f} throughput_samples_s={bs/(sum(ts)/len(ts)):.1f}')
