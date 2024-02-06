import dgl.function as fn
import gc
import torch
import datetime
import copy
import gc
from torch.utils.data import DataLoader
from collections import defaultdict
from data import MetapathDataset



# def hg_propagate(new_g, tgt_type, num_hops, max_hops, extra_metapath, echo=False):
#     for hop in range(1, max_hops):
#         reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
#         for etype in new_g.etypes:
#             stype, _, dtype = new_g.to_canonical_etype(etype)

#             for k in list(new_g.nodes[stype].data.keys()):
#                 if len(k) == hop:
#                     current_dst_name = f'{dtype}{k}'
#                     if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
#                       or (hop > num_hops and k not in reserve_heads):
#                         continue
#                     if echo: print(k, etype, current_dst_name)
#                     new_g[etype].update_all(
#                         fn.copy_u(k, 'm'),
#                         fn.mean('m', current_dst_name), etype=etype)

#         # remove no-use items
#         for ntype in new_g.ntypes:
#             if ntype == tgt_type: continue
#             removes = []
#             for k in new_g.nodes[ntype].data.keys():
#                 if len(k) <= hop:
#                     removes.append(k)
#             for k in removes:
#                 new_g.nodes[ntype].data.pop(k)
#             if echo and len(removes): print('remove', removes)
#         gc.collect()

#         if echo: print(f'-- hop={hop} ---')
#         for ntype in new_g.ntypes:
#             for k, v in new_g.nodes[ntype].data.items():
#                 if echo: print(f'{ntype} {k} {v.shape}')
#         if echo: print(f'------\n')

#     return new_g

def hg_propagate(new_g, tgt_type, num_hops, max_hops, extra_metapath, echo=False):
    for hop in range(1, max_hops):
        reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            for k in list(new_g.nodes[stype].data.keys()):
                meta_path_len = len(k.split('<'))
                # meta_path_len = len(k)
                if meta_path_len == hop:
                    current_dst_name = f'{dtype}<-{etype}-{k}'
                    # current_dst_name = f'{dtype}{k}'
                    
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                      or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    new_g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)
        # remove no-use items
        for ntype in new_g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in new_g.nodes[ntype].data.keys():
                meta_path_len = len(k.split('<'))
                # meta_path_len = len(k)
                if meta_path_len <= hop:
                    removes.append(k)
            for k in removes:
                new_g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo: print(f'-- hop={hop} ---')
        for ntype in new_g.ntypes:
            for k, v in new_g.nodes[ntype].data.items():
                if echo: print(f'{ntype} {k} {v.shape}')
        if echo: print(f'------\n')

    return new_g


def gen_agg_func(msg_name: str, att_name: str, float_upper_bound=0.5):
    def agg_with_random_weight(nodes):
        messages = nodes.mailbox[msg_name]
        device = messages.device
        batch_size = messages.size(0)
        neibor_num = messages.size(1)
        weights = torch.ones(batch_size, neibor_num, 1, device=device)
        weights_float = torch.FloatTensor(batch_size, neibor_num, 1).uniform_(-float_upper_bound, float_upper_bound).to(device)
        weights = torch.softmax(weights + weights_float, dim=1)
        return {att_name: torch.sum(messages * weights, dim=1)}
    return agg_with_random_weight


def hg_propagate_random(new_g, tgt_type, num_hops, max_hops, extra_metapath, echo=False, float_upper_bound=0.5):
     
    for hop in range(1, max_hops):
        reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            for k in list(new_g.nodes[stype].data.keys()):
                meta_path_len = len(k.split('>'))
                # meta_path_len = len(k)
                if meta_path_len == hop:
                    current_dst_name = f'{k}-[{etype}]->{dtype}'
                    # current_dst_name = f'{dtype}{k}'
                    
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                      or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    new_g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        gen_agg_func('m', current_dst_name, float_upper_bound), etype=etype)
        # remove no-use items
        for ntype in new_g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in new_g.nodes[ntype].data.keys():
                meta_path_len = len(k.split('>'))
                # meta_path_len = len(k)
                if meta_path_len <= hop:
                    removes.append(k)
            for k in removes:
                new_g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo: print(f'-- hop={hop} ---')
        for ntype in new_g.ntypes:
            for k, v in new_g.nodes[ntype].data.items():
                if echo: print(f'{ntype} {k} {v.shape}')
        if echo: print(f'------\n')

    return new_g


def clear_hg(new_g, echo=False):
    if echo: print('Remove keys left after propagation')
    for ntype in new_g.ntypes:
        keys = list(new_g.nodes[ntype].data.keys())
        if len(keys) > 0:
            if echo: print(ntype, keys)
            for k in keys:
                if k != ntype:
                    print(k, ntype)
                    new_g.nodes[ntype].data.pop(k)
    return new_g


def propagate_node_features(g, num_hops, tgt_type, float_upper_bound=0):
    # =======
    # features propagate alongside the metapath
    # =======
    prop_tic = datetime.datetime.now()
    g_ = copy.deepcopy(g)
    g_ = hg_propagate_random(g_, tgt_type, num_hops, num_hops+1, [], echo=False, float_upper_bound=float_upper_bound)
    feats = {}
    keys = list(g_.nodes[tgt_type].data.keys())
    print(f'Involved feat keys {keys}')
    for k in keys:
        feats[k] = g_.nodes[tgt_type].data.pop(k)
    g_ = clear_hg(g_, echo=False)
    del g_
    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()
    
    return feats


def propagate_link_features(g, num_hops, target_etypes, float_upper_bound=0):
    prop_tic = datetime.datetime.now()
    node_feature_dict = {}
    target_edge_ntypes = {}
    for etype in target_etypes:
        src_ntype, _, dst_ntype = g.to_canonical_etype(str(etype))
        s_dict = dict()
        d_dict = dict()
        node_feature_dict[src_ntype] = s_dict
        node_feature_dict[dst_ntype] = d_dict
        target_edge_ntypes[etype] = src_ntype, dst_ntype
    for ntype in node_feature_dict.keys():
        g_ = copy.deepcopy(g)
        # g_ = hg_propagate(g_, ntype, num_hops, num_hops+1, [], echo=False)
        g_ = hg_propagate_random(g_, ntype, num_hops, num_hops+1, [], echo=False, float_upper_bound=float_upper_bound)
        for k in g_.nodes[ntype].data.keys():
            node_feature_dict[ntype][k] = g_.nodes[ntype].data[k]
        del g_
    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()
    
    return node_feature_dict, target_edge_ntypes


def gen_processed_node_feats(g, num_hops, tgt_type, init_labels, batch_size, float_upper_bound=0, dl_kwargs={}):
    feats = propagate_node_features(g, num_hops, tgt_type, float_upper_bound)
    dataloader = DataLoader(MetapathDataset(feats, init_labels), batch_size=batch_size, **dl_kwargs)
    processed_feats = defaultdict(list)
    labels = []
    for batch_feats, batch_labels in dataloader:
        for k, v in batch_feats.items():
            processed_feats[k].append(v)
        labels.append(batch_labels)
    processed_feats = {k: torch.cat(v, dim=0) for k, v in processed_feats.items()}
    labels = torch.cat(labels, dim=0)
    del dataloader, feats
    gc.collect()
    return processed_feats, labels


def gen_processed_link_feats(g, num_hops, target_etypes, float_upper_bound=0, dl_kwargs={}):
    node_feature_dict, target_edge_ntypes = propagate_link_features(g, num_hops, target_etypes, float_upper_bound)
    
    all_processed_feats = {}
    for ntype, feat_dict in node_feature_dict.items():
        init_labels = torch.zeros(feat_dict[ntype].shape[0])
        dataloader = DataLoader(MetapathDataset(feat_dict, init_labels), **dl_kwargs)
        processed_feats = defaultdict(list)
        for batch_feats, _ in dataloader:
            for k, v in batch_feats.items():
                processed_feats[k].append(v)
        processed_feats = {k: torch.cat(v, dim=0) for k, v in processed_feats.items()}
        del dataloader, feat_dict
        gc.collect()
        all_processed_feats[ntype] = processed_feats
    return all_processed_feats, target_edge_ntypes