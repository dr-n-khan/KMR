import torch

def threshold_sensitive_loss(pred,target,tau=.3):
    chi=pred-target; sq=chi.square()
    return torch.where(chi.abs()<tau,1.5*sq,2.5*sq-tau*tau).mean()

def alignment_loss(z,prior): return (z-prior).square().sum(dim=-1).mean()

def attention_sparsity(alpha):
    # Paper Eq. 27: L1 norm of attention weights. For softmax-normalized attention this can be constant;
    # retained literally for manuscript fidelity.
    return alpha.abs().sum(dim=-1).mean()
