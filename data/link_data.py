import os
import torch
import numpy as np
import dgl
from torch_sparse import SparseTensor
from pathlib import Path
from tqdm.auto import tqdm
from ogb.linkproppred import Evaluator
from sklearn.metrics import f1_score, auc, roc_auc_score, precision_recall_curve
from dgl import transforms as T
from collections import defaultdict


def evaluate_hgb(edge_list, confidence, labels):
        """
        :param edge_list: shape(2, edge_num)
        :param confidence: shape(edge_num,)
        :param labels: shape(edge_num,)
        :return: dict with all scores we need
        """
        confidence = np.array(confidence)
        labels = np.array(labels)
        roc_auc = roc_auc_score(labels, confidence)
        mrr_list, cur_mrr = [], 0
        t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
        for i, h_id in enumerate(edge_list[0]):
            t_dict[h_id].append(edge_list[1][i])
            labels_dict[h_id].append(labels[i])
            conf_dict[h_id].append(confidence[i])
        for h_id in t_dict.keys():
            conf_array = np.array(conf_dict[h_id])
            rank = np.argsort(-conf_array)
            sorted_label_array = np.array(labels_dict[h_id])[rank]
            pos_index = np.where(sorted_label_array == 1)[0]
            if len(pos_index) == 0:
                continue
            pos_min_rank = np.min(pos_index)
            cur_mrr = 1 / (1 + pos_min_rank)
            mrr_list.append(cur_mrr)
        mrr = np.mean(mrr_list)

        return {'roc_auc': roc_auc, 'MRR': mrr}


def load_hgb_link(dataset_name='amazon', data_path='dataset', 
                 reverse=True, self_loop=True, emb_path=None, embed_size=256):
    data_path = Path(data_path)
    graphs, edge_dict = dgl.load_graphs(str(data_path / f'HGB_{dataset_name}' / f'{dataset_name}.pkl'))
    g = graphs[0]
    target_edge_types = edge_dict['target_edge_types'].numpy().tolist()
    train_edges_pos = edge_dict['all_train']
    val_edges_pos = edge_dict['all_valid']
    test_edges = edge_dict['all_test_with_labels']
    
    for ntype in g.ntypes:
        inp = g.nodes[ntype].data.pop('inp')
        if inp.shape[0] != inp.shape[1]:
            g.nodes[ntype].data[ntype] = inp
        else:
            g.nodes[ntype].data[ntype] = torch.Tensor(g.num_nodes(ntype), embed_size).uniform_(-0.5, 0.5)
    
    if emb_path:
        print(f'Use extra embeddings generated with the {emb_path}')
        for ntype in g.ntypes:
            if 'feat' not in g.nodes[ntype].data.keys():
                emb = torch.load(os.path.join(emb_path, f'{ntype}.pt'),
                                 map_location=torch.device('cpu')).float()
                g.nodes[ntype].data['feat'] = emb

    for ntype in g.ntypes:
        print(ntype, g.nodes[ntype].data[ntype].shape)
    
    if reverse:
        transform = T.Compose([
            T.AddReverse(),
            T.ToSimple()
        ])
        g = transform(g)
    if self_loop:
        transform = T.Compose([
            T.RemoveSelfLoop(), 
            T.AddSelfLoop()
        ])
        g = transform(g)
    
    ones = torch.ones(train_edges_pos.shape[-1]).unsqueeze(0)
    train_edges_pos = torch.vstack([train_edges_pos, ones])
    train_mask = torch.ones(train_edges_pos.shape[-1])

    ones = torch.ones(val_edges_pos.shape[-1]).unsqueeze(0)
    val_edges_pos = torch.vstack([val_edges_pos, ones])
    val_mask = torch.ones(val_edges_pos.shape[-1]) * 0
    test_mask = torch.ones(test_edges.shape[-1]) * -1
    # print(train_edges_pos.shape, val_edges_pos.shape, test_edges.shape)
    
    all_edges = torch.cat([train_edges_pos, val_edges_pos, test_edges], dim=-1)
    all_mask = torch.cat([train_mask, val_mask, test_mask])
    # print(all_edges.shape, all_mask.shape)
    
    src_dict = {}
    dst_dict = {}
    mask_dict = {}
    label_dict = {}
    for target_etype in target_edge_types:
        e_mask = (all_edges[0] == target_etype)
        edges = all_edges[1:3, e_mask]
        src_dict[target_etype] = edges[0].long()
        dst_dict[target_etype] = edges[1].long()

        train_mask = (all_mask == 1)[e_mask]
        val_mask = (all_mask == 0)[e_mask]
        test_mask = (all_mask == -1)[e_mask]
        train_eid = (train_mask == 1).nonzero(as_tuple=True)[0]
        val_eid = (val_mask == 1).nonzero(as_tuple=True)[0]
        test_eid = (test_mask == 1).nonzero(as_tuple=True)[0]
        mask_dict[target_etype] = (train_eid, val_eid, test_eid)
        label_dict[target_etype] = all_edges[3, e_mask]
    
    return g, target_edge_types, src_dict, dst_dict, mask_dict, label_dict, evaluate_hgb