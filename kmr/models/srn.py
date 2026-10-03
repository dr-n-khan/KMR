import math, collections, torch

def srn_filter(edges, n_entities, threshold=.30, hops=5):
    """Practical SRN-style relevance filtering based on manuscript Eqs. 6,8,9,10,11.
    Interaction-specific temporal tendency terms require source timestamps not supplied by the paper dataset interface;
    this implementation uses neighborhood overlap as the available tendency proxy and documents that assumption.
    """
    adj=[set() for _ in range(n_entities)]
    for h,_,t in edges.tolist(): adj[h].add(t); adj[t].add(h)
    scored=[]
    for h,r,t in edges.tolist():
        inter=len(adj[h]&adj[t]); union=max(1,len(adj[h]|adj[t])); jacc=inter/union
        adamic=sum(1/max(math.log(max(2,len(adj[y]))),1e-8) for y in (adj[h]&adj[t]))
        tendency=(inter+1)/(union+1)
        score=tendency+0.5*adamic+0.5*jacc
        scored.append((score,h,r,t))
    if not scored: return edges
    vals=torch.tensor([x[0] for x in scored],dtype=torch.float32); cutoff=torch.quantile(vals, min(max(threshold,0.),1.))
    keep=[[h,r,t] for s,h,r,t in scored if s>=float(cutoff)]
    return torch.tensor(keep,dtype=torch.long) if keep else edges
