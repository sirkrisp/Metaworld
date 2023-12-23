import numpy as np
import torch


def to_numpy(t):
    if isinstance(t, dict):
        for k, v in t.items():
            t[k] = to_numpy(v)
        return t
    elif isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    else:
        return t
    
def to_torch(t, device="cuda"):
    if isinstance(t, dict):
        for k, v in t.items():
            t[k] = to_torch(v, device)
        return t
    elif isinstance(t, np.ndarray):
        return torch.from_numpy(t).to(device)
    else:
        return t.to(device)