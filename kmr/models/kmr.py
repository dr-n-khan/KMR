import torch
from torch import nn
from .losses import threshold_sensitive_loss, alignment_loss, attention_sparsity

class RelGraphEncoder(nn.Module):
    def __init__(self,n_entities,n_relations,d):
        super().__init__(); self.ent=nn.Embedding(n_entities,d); self.rel=nn.Parameter(torch.empty(n_relations,d,d)); nn.init.xavier_uniform_(self.ent.weight); nn.init.xavier_uniform_(self.rel)
    def forward(self,edges):
        base=self.ent.weight; out=torch.zeros_like(base); deg=torch.zeros(base.size(0),device=base.device)
        if edges.numel()==0:return torch.relu(base)
        h,r,t=edges.T; msg=torch.bmm(self.rel[r],base[t].unsqueeze(-1)).squeeze(-1); out.index_add_(0,h,msg); deg.index_add_(0,h,torch.ones_like(h,dtype=torch.float)); out=out/deg.clamp_min(1).unsqueeze(1)
        return torch.relu(out)

class KMR(nn.Module):
    def __init__(self,n_users,n_items,n_entities,n_relations,user_dim,item_dim,cfg):
        super().__init__(); d=cfg['embedding_dim']; hid=cfg.get('hidden_dim',d); self.cfg=cfg
        self.graph=RelGraphEncoder(n_entities,n_relations,d)
        self.user_raw=nn.Embedding(n_users,d); self.user_feat=nn.Linear(user_dim,d); self.user_context=nn.Linear(d,d)
        self.gate=nn.Linear(d*3,3); self.user_bind=nn.Linear(d*3,d)
        self.item_attr=nn.Linear(item_dim,d); self.item_attention=nn.Linear(d,1); self.item_bind=nn.Linear(d,d)
        self.interaction=nn.Bilinear(d,d,1); self.pred=nn.Linear(1,1); self.drop=nn.Dropout(cfg.get('dropout',.5)); self.act=nn.ReLU()
        self.user_prior=nn.Embedding(n_users,d); self.item_prior=nn.Embedding(n_items,d)
        nn.init.zeros_(self.user_prior.weight); nn.init.zeros_(self.item_prior.weight)
    def representations(self,user_features,item_features,edges,user_entity,item_entity,disable=None):
        disable=set(disable or []); ge=self.graph(edges)
        raw=self.user_raw.weight; uf=self.act(self.user_feat(user_features)); gc=ge[user_entity]
        # Opinion adjustment: u~ = u + lambda(alpha*c + beta*g). Here normalized user features proxy c.
        if 'UA' not in disable:
            raw=raw+self.cfg.get('opinion_lambda',.3)*(self.cfg.get('opinion_alpha',.5)*uf+self.cfg.get('opinion_beta',.5)*gc)
        parts=torch.cat([raw,uf,gc],1); gates=torch.softmax(self.gate(parts),1)
        if 'RB' in disable: zu=(raw+uf+gc)/3
        else: zu=self.act(self.user_bind(torch.cat([gates[:,0:1]*raw,gates[:,1:2]*uf,gates[:,2:3]*gc],1)))
        # Treat each numeric MDP feature as an attribute; shared encoder + feature attention.
        x=item_features; attr=self.act(self.item_attr(x)); alpha=torch.ones((x.size(0),1),device=x.device)
        zi=self.act(self.item_bind(attr)) if 'RB' not in disable else attr
        return zu,zi,alpha
    def forward(self,u,i,user_features,item_features,edges,user_entity,item_entity,disable=None):
        zu,zi,a=self.representations(user_features,item_features,edges,user_entity,item_entity,disable); x=self.interaction(self.drop(zu[u]),self.drop(zi[i])); return self.act(self.pred(x)).squeeze(-1),zu,zi,a
    def objective(self,pred,target,zu,zi,alpha):
        c=self.cfg; rec=threshold_sensitive_loss(pred,target,c.get('tau',.3)); al=alignment_loss(zu,self.user_prior.weight)+alignment_loss(zi,self.item_prior.weight); sp=attention_sparsity(alpha); l2=sum(p.square().sum() for p in self.parameters())
        total=rec+c.get('lambda1',.5)*al+c.get('lambda2',.45)*sp+c.get('lambda3',1e-7)*l2
        return total,{'rec':rec.detach(),'align':al.detach(),'sparse':sp.detach()}
