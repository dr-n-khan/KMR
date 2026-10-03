from kmr.data.synthetic import make_demo
from kmr.data.loader import load_bundle,user_level_split

def test_demo(tmp_path):
 make_demo(tmp_path,30,20,1);b=load_bundle(tmp_path);tr,va,te=user_level_split(b['interactions'],seed=1);assert len(tr)>0 and b['n_relations']>0
