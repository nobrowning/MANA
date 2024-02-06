import random
import numpy as np
import torch
import dgl

from .propagate import hg_propagate, hg_propagate_random, clear_hg
from .propagate import gen_processed_node_feats, gen_processed_link_feats



def set_random_seed(seed:int=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
        
def get_n_params(model: torch.nn.Module):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
