from pathlib import Path
import numpy as np, pandas as pd

def make_demo(out, users=300, items=120, seed=42):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(seed)
    stages=np.array(["early","moderate","severe"])
    udf=pd.DataFrame({"user_id":[f"u{i}" for i in range(users)],"age":rng.integers(55,91,users),"sex":rng.integers(0,2,users),"stage":rng.choice(stages,users,p=[.32,.40,.28])})
    for j in range(5): udf[f"context_{j}"]=rng.normal(size=users)
    idf=pd.DataFrame({"item_id":[f"m{i}" for i in range(items)],"item_type":rng.integers(0,2,items)})
    for j in range(7): idf[f"feature_{j}"]=rng.normal(size=items)
    rows=[]
    for u in range(users):
        n=int(rng.integers(6,min(25,items)))
        for it in rng.choice(items,n,replace=False):
            y=float(rng.random()>.20)
            rows.append((f"u{u}",f"m{it}",y,y,float(rng.random()>.65),"helpful care experience" if y else "not helpful"))
    pd.DataFrame(rows,columns=["user_id","item_id","label","explicit","implicit","review"]).to_csv(out/'interactions.csv',index=False)
    rels=["treats","causes","associated_with","recommended_for","contraindicated_with","belongs_to","interacts_with"]
    entities=list(udf.user_id)+list(idf.item_id)+[f"c{i}" for i in range(80)]
    kg=[]
    for _ in range(max(2000,users*8)):
        h,t=rng.choice(entities,2,replace=False); kg.append((h,rng.choice(rels),t))
    pd.DataFrame(kg,columns=["head","relation","tail"]).to_csv(out/'kg.csv',index=False)
    udf.to_csv(out/'users.csv',index=False); idf.to_csv(out/'items.csv',index=False)
