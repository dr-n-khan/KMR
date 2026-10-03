from pathlib import Path
import yaml, random, numpy as np, torch

def load_config(path):
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_from_config(name="auto"):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))
