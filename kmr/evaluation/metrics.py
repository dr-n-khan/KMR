import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def ndcg_at_k(y,s,k):
    o=np.argsort(-s)[:k]; rel=np.asarray(y)[o]; dcg=np.sum((2**rel-1)/np.log2(np.arange(2,len(rel)+2))); ideal=np.sort(np.asarray(y))[::-1][:k]; idcg=np.sum((2**ideal-1)/np.log2(np.arange(2,len(ideal)+2))); return float(dcg/idcg) if idcg else 0.
def ranking_by_user(frame,scores,ks):
    f=frame.copy(); f['score']=scores; out={f'NDCG@{k}':[] for k in ks}
    for _,g in f.groupby('user_idx'):
        for k in ks: out[f'NDCG@{k}'].append(ndcg_at_k(g.label.values,g.score.values,k))
    return {k:float(np.mean(v)) if v else 0. for k,v in out.items()}
def errors(y,s): return {'RMSE':float(mean_squared_error(y,s)**.5),'MAE':float(mean_absolute_error(y,s))}
def coverage(recs,n_items): return len(set(x for row in recs for x in row))/max(1,n_items)
def novelty(recs,pop,n_inter):
    vals=[-np.log2(max(pop.get(x,1),1)/max(n_inter,1)) for row in recs for x in row]; return float(np.mean(vals)) if vals else 0.
def diversity(recs,item_vectors):
    vals=[]
    for row in recs:
        for a in range(len(row)):
            for b in range(a+1,len(row)):
                x=item_vectors[row[a]]; y=item_vectors[row[b]]; den=np.linalg.norm(x)*np.linalg.norm(y); vals.append(1-float(x@y/den) if den else 0.)
    return float(np.mean(vals)) if vals else 0.
