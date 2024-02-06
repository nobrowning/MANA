import gc
import time
import argparse
import datetime
import torch
from typing import Dict, Tuple, List, Any, Set
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import model as model_lib
from utils import gen_processed_link_feats, set_random_seed, get_n_params
from data import load_hgb_link, LinkDataset
import logging
import wandb
import os
import copy
from collections import defaultdict
from tqdm.auto import tqdm, trange


def train(model, node_feats, src_ntype, dst_ntype, data_loader, loss_func, optimizer, lr_scheduler, device, scalar=None):
    model.train()
    data_loader.dataset.dataset.gen_neg = True
    src_feat_dict = {k: v.to(device) for k, v in node_feats[src_ntype].items()}
    dst_feat_dict = {k: v.to(device) for k, v in node_feats[dst_ntype].items()}
    total_loss = 0
    iter_num = 0
    data_iter = iter(data_loader)
    n_batch = len(data_loader)
    for batch_idx in trange(n_batch, dynamic_ncols=True):
        batch_src_idx, batch_dst_idx, batch_labels = next(data_iter)
        batch_src_idx = batch_src_idx.to(device)
        batch_dst_idx, batch_labels = batch_dst_idx.flatten().to(device), batch_labels.flatten().to(device)
        
        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                scores = model(src_feat_dict, dst_feat_dict, batch_src_idx, batch_dst_idx)
                loss_train = loss_func(scores, batch_labels)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
            lr_scheduler.step()
        else:
            scores = model(src_feat_dict, dst_feat_dict, batch_src_idx, batch_dst_idx)
            loss_train = loss_func(scores, batch_labels)
            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()
        
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    return loss


@torch.no_grad()
def eval(model, node_feats, src_ntype, dst_ntype, data_loader, loss_func, evaluator, device, edge_list=None):
    model.eval()
    src_feat_dict = {k: v.to(device) for k, v in node_feats[src_ntype].items()}
    dst_feat_dict = {k: v.to(device) for k, v in node_feats[dst_ntype].items()}
    total_loss = 0
    iter_num = 0
    score_list = []
    label_list = []
    data_iter = iter(data_loader)
    n_batch = len(data_loader)
    for batch_idx in trange(n_batch, dynamic_ncols=True):
        batch_src_idx, batch_dst_idx, batch_labels = next(data_iter)
        batch_src_idx = batch_src_idx.to(device)
        batch_dst_idx, batch_labels = batch_dst_idx.flatten().to(device), batch_labels.flatten().to(device)
        
        scores = model(src_feat_dict, dst_feat_dict, batch_src_idx, batch_dst_idx)
        loss_train = loss_func(scores, batch_labels)

        score_list.append(scores)
        label_list.append(batch_labels)
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    score_list = torch.cat(score_list, dim=0).detach()
    label_list = torch.cat(label_list, dim=0)
    
    if edge_list is not None:
        score_list = torch.sigmoid(score_list)
        eval_result = evaluator(edge_list.numpy(), score_list.cpu().numpy(), label_list.cpu().numpy())
    else:
        eval_result = None
    
    return loss, eval_result


def main(args):
    # if args.seed > 0:
    #     set_random_seed(args.seed)
    g, target_etypes, src_dict, dst_dict, mask_dict, label_dict, evaluator = \
    load_hgb_link(
        dataset_name=args.dataset, 
        data_path=args.data_path, 
        reverse=args.link_reverse, 
        self_loop=args.self_loop, 
        emb_path=args.emb_path, 
        embed_size=args.embed_size
    )
    
    dl_kwargs = {'num_workers': args.workers, 'batch_size': args.batch_size, 
                     'persistent_workers': True if args.workers > 0 else False,
                     'pin_memory': True}
    
    all_processed_feats, target_edge_ntypes = \
        gen_processed_link_feats(g, args.num_hops, target_etypes, float_upper_bound=0, dl_kwargs=dl_kwargs)

    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    device = "cuda:{}".format(args.gpu) if not args.cpu else 'cpu'
    # device = "cpu"
    
    data_loader_dict = defaultdict(dict)
    eval_edge_dict = defaultdict(dict)
    for etype in target_etypes:
        src_ntype, _, dst_ntype = g.to_canonical_etype(str(etype))
        src_ntype, dst_ntype = target_edge_ntypes[etype]
        train_eid, val_eid, test_eid = mask_dict[etype]
        edge_list = torch.vstack([src_dict[etype], dst_dict[etype]])
        eval_edge_dict['test'][etype] = edge_list[:, test_eid]
        eval_edge_dict['val'][etype] = edge_list[:, val_eid]
        
        full_dataset = LinkDataset(src_dict[etype], dst_dict[etype], 
                                   label_dict[etype],
                                   g.num_nodes(dst_ntype), device)
        data_loader_dict[etype]['train'] = DataLoader(Subset(full_dataset, train_eid), shuffle=True, **dl_kwargs)
        data_loader_dict[etype]['val'] = DataLoader(Subset(full_dataset, val_eid), shuffle=True, **dl_kwargs)
        data_loader_dict[etype]['test'] = DataLoader(Subset(full_dataset, test_eid), shuffle=False, **dl_kwargs)
    
    src_data_size = {k: v.size(-1) for k, v in all_processed_feats[src_ntype].items()}
    dst_data_size = {k: v.size(-1) for k, v in all_processed_feats[dst_ntype].items()}
    
    # if args.n_batch is None or args.n_batch > len(data_loader_dict[etype]['train']):
    #     args.n_batch = len(data_loader_dict[etype]['train'])
    
    
    
    # label_emb = torch.zeros((num_nodes, n_classes)).to(device)
    # =======
    # Construct network
    # =======
    model_class = getattr(model_lib, args.model)
    data_args = model_lib.DataArgs(src_data_size, src_ntype, 
                         len(all_processed_feats[src_ntype]), 
                         dst_data_size, dst_ntype, 
                         len(all_processed_feats[dst_ntype]))
    model = model_class(dataset=args.dataset,
                           data_args=data_args,
                           nfeat=args.embed_size,
                           hidden=args.hidden,
                           nclass=args.hidden, # 1
        dropout=args.dropout,
        input_drop=args.input_drop,
        att_drop=args.att_drop,
        n_layers=args.n_layers_1,
        act=args.act,
        residual=args.residual,
        bns=args.bns,
        n_trans=args.n_trans
    )
    
    model = model.to(device)
    # logging.info(model)
    logging.info(f"# Params: {get_n_params(model)}")
    
#     evaluator = LinkEvaluator().to(device)
    # evaluator = Evaluator(n_classes, device)

    loss_func = nn.BCEWithLogitsLoss()
    # total_steps = args.n_batch * args.epochs
    total_steps = sum([len(data_loader_dict[etype]['train']) for etype in target_edge_ntypes.keys()]) * args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10,\
                        max_lr=args.lr, total_steps=total_steps)
    # lr_scheduler = None

    best_epoch = 0
    best_mrr_val = 0
    best_roc_val = 0
    best_mrr_test = 0
    best_roc_test = 0
    best_val_loss = 999
    count = 0

    for epoch in range(args.epochs):
        gc.collect()
        torch.cuda.empty_cache()
        start = time.time()
        losses = 0
        for etype in target_etypes:
            src_ntype, dst_ntype = target_edge_ntypes[etype]
            loss = train(model, all_processed_feats, src_ntype, dst_ntype,
                         data_loader_dict[etype]['train'], loss_func, optimizer, lr_scheduler, device, scalar=scalar)
            losses +=loss
        end = time.time()
        losses = losses / len(target_etypes)
        log = "Epoch {}, Time(s): {:.4f}, Train loss {:.4f} ".format(epoch, end-start, losses)
        # torch.cuda.empty_cache()
        wandb.log({"Loss/train": losses}, step=epoch)

        if epoch % args.eval_every == 0:
            val_losses, test_losses = 0, 0
            all_eval_scores = defaultdict(list)
            for etype in target_etypes:
                full_dataset.gen_neg = True
                src_ntype, dst_ntype = target_edge_ntypes[etype]
                src_ntype, dst_ntype = target_edge_ntypes[etype]
                loss_val, eval_scores_val = eval(model, all_processed_feats, src_ntype, dst_ntype,
                                   data_loader_dict[etype]['val'], loss_func, evaluator, device, eval_edge_dict['val'][etype])
                full_dataset.gen_neg = False
                loss_test, eval_scores_test = eval(model, all_processed_feats, src_ntype, dst_ntype, 
                                              data_loader_dict[etype]['test'], loss_func, evaluator, device, eval_edge_dict['test'][etype])
                for k, v in eval_scores_val.items():
                    all_eval_scores[k + '/val'].append(v)
                for k, v in eval_scores_test.items():
                    all_eval_scores[k + '/test'].append(v)
                val_losses += loss_val
                test_losses += loss_test
            val_losses = val_losses / len(target_etypes)
            test_losses = test_losses / len(target_etypes)
            all_eval_scores = {k: sum(v)/len(target_etypes) for k, v in all_eval_scores.items()}
            
            log += f'Val loss: {val_losses:.4f}, Test loss: {test_losses:.4f}\n'
            log += 'roc_auc/val: {:.4f}, MRR/val: {:.4f}, '.format(all_eval_scores['roc_auc/val'], all_eval_scores['MRR/val'])
            log += 'roc_auc/test: {:.4f}, MRR/test: {:.4f}\n'.format(all_eval_scores['roc_auc/test'], all_eval_scores['MRR/test'])
            

            # if all_eval_scores['roc_auc/val'] > best_roc_val:
            if val_losses < best_val_loss:
                best_val_loss = val_losses
                best_epoch = epoch
                best_roc_val = all_eval_scores['roc_auc/val']
                best_mrr_val = all_eval_scores['MRR/val']
                best_roc_test = all_eval_scores['roc_auc/test']
                best_mrr_test = all_eval_scores['MRR/test']
                count = 0
                if args.save_model:
                    torch.save(model.state_dict(), args.save_path)
            else:
                count = count + args.eval_every
                if count >= args.patience:
                    break
            log += "Best Epoch {}, roc_auc/val {:.4f}, mrr/val {:.4f}, ".format(best_epoch, best_roc_val, best_mrr_val)
            log += "roc_auc/test {:.4f}, mrr/test {:.4f} ".format(best_roc_test, best_mrr_test)
        logging.info(log)
        wandb.log({
                "Best_Epoch": best_epoch,
                "mrr/val": all_eval_scores['MRR/val'],
                "roc/val": all_eval_scores['roc_auc/val'],
                "mrr/test": all_eval_scores['MRR/test'],
                "roc/test": all_eval_scores['roc_auc/test'],
                "best_mrr/val": best_mrr_val,
                "best_roc/val": best_roc_val,
                "best_mrr/test": best_mrr_test,
                "best_roc/test": best_roc_test
            }, step=epoch)

    logging.info("Best Epoch {}, mrr {:.4f}, roc {:.4f} ".format(best_epoch, best_mrr_test, best_roc_test))
    wandb.log({
        "Best_MRR/test": best_mrr_test,
        "Best_ROC/test": best_roc_test,
        "Best_Epoch": best_epoch
    })


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='SeHGNN')
    ## For environment costruction
    parser.add_argument("--seeds", nargs='+', type=int, default=[1],
                        help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="LastFM")
    parser.add_argument("--model", type=str, default='SimpleLink', help='model name')
    parser.add_argument("--link-reverse", action='store_false', default=True)
    parser.add_argument("--self-loop", action='store_false', default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--data-path", type=str, default='dataset/')
    parser.add_argument("--epochs", type=int, default=300,
                        help="The epoch setting.")
    parser.add_argument('--n-batch', type=int)
    ## For pre-processing
    parser.add_argument("--emb-path", type=str)
    parser.add_argument("--embed-size", type=int, default=256,
                        help="inital embedding size of nodes with no attributes")
    parser.add_argument("--num-hops", type=int, default=2,
                        help="number of hops for propagation of raw labels")
    ## For network structure
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--n-layers-1", type=int, default=2,
                        help="number of layers of feature projection")
    parser.add_argument("--n-layers-2", type=int, default=2,
                        help="number of layers of the downstream task")
    parser.add_argument("--n-layers-3", type=int, default=4,
                        help="number of layers of residual label connection")
    parser.add_argument("--n-trans", type=int, default=1,
                        help="number of layers of transformer")
    parser.add_argument("--input-drop", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.,
                        help="attention dropout of model")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to connect the input features")
    parser.add_argument("--act", type=str, default='relu',
                        help="the activation function of the model")
    parser.add_argument("--bns", action='store_true', default=False,
                        help="whether to process the input features")
    ## for training
    parser.add_argument("--amp", action='store_true', default=False,
                        help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument('--workers', type=int, default=4, help='number of data loader workers') 
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--gama", type=float, default=0.5,
                        help="parameter for the KL loss")
    parser.add_argument('--logsubfix', type=str, default='', help='logsubfix')
    parser.add_argument('--save-model', default=False, action='store_true')
    parser.add_argument('--save-path', help='save path', default='checkpoints/cls_')
    parser.add_argument('--off-wandb', default=False, action='store_true')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    tags = [str(args.model)]
    if len(args.logsubfix) != 0:
        [tags.append(t) for t in args.logsubfix.split(',')]
    run = wandb.init(project='MANA', entity='nobody', tags=tags, mode="disabled" if args.off_wandb else None)
    wandb.config.update(args)
    config = wandb.config
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler("logs/simple_link_{}_{}.log".format(
                                args.logsubfix, run.id)),
                            logging.StreamHandler()
                        ],
                        level=logging.INFO)
    logging.info(args)
    args.save_path = args.save_path + '{}_{}.pt'.format(args.logsubfix, run.id)
    run.log_code()

    for seed in args.seeds:
        args.seed = seed
        logging.info('Restart with seed = {}'.format(seed))
        main(args)
