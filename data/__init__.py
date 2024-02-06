from .link_data import load_hgb_link
from torch.utils.data import Dataset
from typing import Dict
from collections import defaultdict
import torch
import numpy as np
from torch.utils.data import DataLoader
import gc


        

class MetapathDataset(Dataset):
    def __init__(self, feats: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.feats = feats
        self.labels = labels
        
    def __getitem__(self, index):
        feat_dict = {
            k: v[index] for k, v in self.feats.items()
        }
        label = self.labels[index]
        
        return feat_dict, label
    
    def __len__(self):
        return self.labels.shape[0]
  

class NodeIdDataset(Dataset):
    def __init__(self, node_ids, neg_tensor: torch.Tensor, num_neg: int):
        self.node_ids = node_ids
        self.neg_tensor = neg_tensor
        self.num_neg = num_neg
        self.return_pair = True
        
    def __getitem__(self, index):
        node_id = self.node_ids[index]
        if self.return_pair:
            neg_ids = self.neg_tensor[index]
            sample_idxs = torch.randperm(self.neg_tensor.shape[1])[:self.num_neg]
            neg_ids = neg_ids[sample_idxs]
            return node_id, neg_ids
        else:
            return node_id
    
    def __len__(self):
        return len(self.node_ids)


def gen_neg_nodes(w2v_model, node_ids, num_negs):
    all_neg_nids = []
    
    if type(node_ids) == torch.Tensor:
        node_ids = node_ids.numpy().tolist()
    
    for nid in node_ids:
        neg_nids = w2v_model.wv.similar_by_key(str(nid), topn=num_negs)
        neg_nids = [int(neg_nid[0]) for neg_nid in neg_nids]
        all_neg_nids.append(neg_nids)
    all_neg_nids = torch.LongTensor(all_neg_nids)
    return all_neg_nids


class EmbeddingDataset(Dataset):
    def __init__(self, feats: torch.Tensor, labels: torch.Tensor):
        self.feats = feats
        self.labels = labels
        
    def __getitem__(self, index):
        return self.feats[index], self.labels[index]
    
    def __len__(self):
        return self.labels.shape[0]


class LinkMetapathDataset(Dataset):
    def __init__(self, 
                 src_feats: Dict[str, torch.Tensor],
                 dst_feats: Dict[str, torch.Tensor],
                 src_list: torch.Tensor,
                 dst_list: torch.Tensor,
                 labels: torch.Tensor,
                 num_dst: int, gen_neg=False, device='cpu'):
        self.src_feats = src_feats
        self.dst_feats = dst_feats
        self.src_list = src_list
        self.dst_list = dst_list
        self.labels = labels
        self.num_dst = num_dst
        self.gen_neg = gen_neg
        self.device = device
    
    def __get_dict_feature__(self, input_dict,index):
        feat_dict = {
            k: v[index] for k, v in input_dict.items()
        }
        return feat_dict
        
    def __getitem__(self, index):
        src_idx = self.src_list[index]
        dst_idx = self.dst_list[index]
        
        if self.gen_neg:
            neg_idx = np.random.randint(self.num_dst, size=1)[0]
            dst_idx = [dst_idx.item(), neg_idx]
            label = torch.FloatTensor([1, 0], device=self.device)
        else:
            label = self.labels[index]
            
        
        src_feat_dict = self.__get_dict_feature__(self.src_feats, src_idx)
        dst_feat_dict = self.__get_dict_feature__(self.dst_feats, dst_idx)
        
        return src_feat_dict, dst_feat_dict, label
    
    def __len__(self):
        return self.src_feats.shape[0]


class LinkDataset(Dataset):
    def __init__(self, 
                 src_list: torch.Tensor,
                 dst_list: torch.Tensor,
                 labels: torch.Tensor,
                 num_dst: int, 
                 gen_neg=False, device='cpu'):
        self.src_list = src_list
        self.dst_list = dst_list
        self.labels = labels
        self.num_dst = num_dst
        self.gen_neg = gen_neg
        self.device = device
        
    def __getitem__(self, index):
        src_idx = self.src_list[index]
        dst_idx = self.dst_list[index]
        
        if self.gen_neg:
            neg_idx = np.random.randint(self.num_dst, size=1)[0]
            dst_idx = torch.LongTensor([dst_idx.item(), neg_idx])
            label = torch.FloatTensor([1, 0], device=self.device)
        else:
            label = self.labels[index]
        
        return src_idx, dst_idx, label
    
    def __len__(self):
        return len(self.src_list)


def dict_collate_fn(data):
    src_feat_dict = defaultdict(list)
    dst_feat_dict = defaultdict(list)
    all_label = []
    for src_f_dict, dst_f_dict, label in data:
        for k, v in src_f_dict.items():
            src_feat_dict[k].append(v)
        for k, v in dst_f_dict.items():
            dst_feat_dict[k].append(v)
        all_label.append(label)
    
    src_feat_dict = {k: torch.vstack(v) for k, v in src_feat_dict.items()}
    dst_feat_dict = {k: torch.vstack(v) for k, v in dst_feat_dict.items()}
    all_label = torch.stack(all_label).flatten()
    
    return src_feat_dict, dst_feat_dict, all_label
