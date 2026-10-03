import torch
from kmr.models.losses import threshold_sensitive_loss

def test_threshold_loss_finite():
 p=torch.tensor([0.1,0.8]);y=torch.tensor([0.,1.]);assert torch.isfinite(threshold_sensitive_loss(p,y,.3))
