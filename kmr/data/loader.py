from pathlib import Path
import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class InteractionDataset(Dataset):
    def __init__(self, frame): self.f=frame.reset_index(drop=True)
    def __len__(self): return len(self.f)
    def __getitem__(self,i):
        r=self.f.iloc[i]
        return torch.tensor(int(r.user_idx)),torch.tensor(int(r.item_idx)),torch.tensor(float(r.label),dtype=torch.float32)

def _features(df,id_col,exclude):
    cols=[c for c in df.columns if c not in exclude and c!=id_col and pd.api.types.is_numeric_dtype(df[c])]
    if not cols: return np.ones((len(df),1),dtype='float32')
    x=df[cols].fillna(0).to_numpy(dtype='float32')
    return StandardScaler().fit_transform(x).astype('float32')

def load_bundle(data_dir):
    d=Path(data_dir); inter=pd.read_csv(d/'interactions.csv'); users=pd.read_csv(d/'users.csv'); items=pd.read_csv(d/'items.csv'); kg=pd.read_csv(d/'kg.csv')
    req={'user_id','item_id','label'}
    if not req.issubset(inter.columns): raise ValueError(f'interactions.csv needs {req}')
    uids=users.user_id.astype(str).tolist(); iids=items.item_id.astype(str).tolist(); um={x:i for i,x in enumerate(uids)}; im={x:i for i,x in enumerate(iids)}
    inter.user_id=inter.user_id.astype(str); inter.item_id=inter.item_id.astype(str); inter=inter[inter.user_id.isin(um)&inter.item_id.isin(im)].copy()
    inter['user_idx']=inter.user_id.map(um); inter['item_idx']=inter.item_id.map(im)
    uf=_features(users,'user_id',{'stage','health_text'}); itf=_features(items,'item_id',{'item_type'})
    ents=sorted(set(kg['head'].astype(str))|set(kg['tail'].astype(str))|set(uids)|set(iids)); em={e:i for i,e in enumerate(ents)}
    rels=sorted(kg.relation.astype(str).unique()); rm={r:i for i,r in enumerate(rels)}
    edges=np.array([[em[str(h)],rm[str(r)],em[str(t)]] for h,r,t in kg[['head','relation','tail']].itertuples(index=False)],dtype=np.int64)
    user_ent=torch.tensor([em[x] for x in uids]); item_ent=torch.tensor([em[x] for x in iids])
    return dict(interactions=inter, users=users, items=items, kg=kg, user_features=torch.tensor(uf), item_features=torch.tensor(itf), edges=torch.tensor(edges), user_entity=user_ent, item_entity=item_ent, n_users=len(uids),n_items=len(iids),n_entities=len(ents),n_relations=len(rels))

def user_level_split(frame, train=.7,val=.1,seed=42):
    rng=np.random.default_rng(seed); tr=[];va=[];te=[]
    for _,g in frame.groupby('user_idx'):
        idx=np.arange(len(g)); rng.shuffle(idx); n=len(idx); a=max(1,int(n*train)); b=max(a+1,int(n*(train+val))) if n>=3 else a
        tr.append(g.iloc[idx[:a]]); va.append(g.iloc[idx[a:b]]); te.append(g.iloc[idx[b:]])
    cat=lambda xs: pd.concat([x for x in xs if len(x)],ignore_index=True) if any(len(x) for x in xs) else frame.iloc[:0].copy()
    return cat(tr),cat(va),cat(te)
