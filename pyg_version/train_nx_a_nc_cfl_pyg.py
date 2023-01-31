from __future__ import print_function, unicode_literals
from utils.train_utils import BinFullMeter, AverageMeter
from utils.utils import ConfigClass
import pickle as pkl
import json
import os
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from pyg_version.codeflaws.dataloader_cfl_pyg import PyGStatementDataset
from pyg_version.model import MPNNModel_A_T_L
from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset
from nbl.dataloader_cfl import NBLPyGCFLNxStatementDataset
import pandas as pd
from utils.common import device
from utils.data_utils import split_nx_dataset, AstGraphMetadata
import argparse


def train(model, dataloader, n_epochs, eval_func, start_epoch=0, save_dir="save"):
    opt = torch.optim.Adam(model.parameters())
    avg_loss, avg_acc = AverageMeter(), AverageMeter()

    top_1_rec, top_2_rec, top_5_rec, top_10_rec = [
        AverageMeter() for _ in range(4)
    ]
    f1_meter = BinFullMeter()
    best_f1, best_top1, best_top2, best_top5, best_top10 = [0.0] * 5
    best_f1_train, best_top1_train, best_top2_train, best_top5_train, \
            best_top10_train = [0.0] * 5
    for epoch in range(n_epochs):
        model.train()
        for meter in [
                avg_loss, avg_acc, f1_meter, top_10_rec, top_5_rec, top_2_rec,
                top_1_rec
        ]:
            meter.reset()

        bar = tqdm.trange(len(dataloader))
        bar.set_description(f'Epoch {epoch}')
        for i in bar:
            g, stmt_nodes = dataloader[i]
            if g is None:
                continue
            g, stmt_nodes = g.to(device), stmt_nodes.to(device)
            ast_lb = g.lbl[stmt_nodes]
            non_zeros_lbs = torch.nonzero(ast_lb).detach()
            ast_lbidxs = torch.flatten(non_zeros_lbs).detach().cpu().tolist()
            logits, preds = model(g.xs, g.ess)
            loss = F.cross_entropy(logits[stmt_nodes], ast_lb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ast_lb = ast_lb.detach().cpu()
            if non_zeros_lbs.shape[0] == 0:
                continue
            _, ast_cal = torch.max(preds[stmt_nodes].detach().cpu(), dim=1)
            sus_preds = preds[stmt_nodes, 1].detach().cpu()
            top_updates = [(10, top_10_rec), (5, top_5_rec), (2, top_2_rec),
                           (1, top_1_rec)]
            for k, meter in top_updates:
                k = min(stmt_nodes.shape[0], k)
                _, indices = torch.topk(sus_preds, k)
                topk_val = indices[:k].tolist()
                meter.update(int(any([i in ast_lbidxs for i in topk_val])), 1)
            avg_loss.update(loss.item(), stmt_nodes.shape[0])
            avg_acc.update(
                torch.sum(ast_cal.cpu() == ast_lb.cpu()).item() /
                stmt_nodes.shape[0], stmt_nodes.shape[0])
            f1_meter.update(ast_cal, ast_lb)
            bar.set_postfix(ast_loss=loss.item(), acc=avg_acc.avg)

        out_dict = {
            'top_1': top_1_rec.avg,
            'top_2': top_2_rec.avg,
            'top_5': top_5_rec.avg,
            'top_10': top_10_rec.avg,
            'mean_acc': avg_acc.avg,
            'mean_loss': avg_loss.avg,
            'mean_ast_acc': avg_acc.avg,
            'mean_ast_loss': avg_loss.avg,
            'f1': f1_meter.get()
        }

        best_top1_train = max(top_1_rec.avg, best_top1_train)
        best_top2_train = max(top_2_rec.avg, best_top2_train)
        best_top5_train = max(top_5_rec.avg, best_top5_train)
        best_top10_train = max(top_10_rec.avg, best_top10_train)
        if f1_meter.get()['aux_f1'] != 'unk':
            best_f1_train = max(best_f1_train, f1_meter.get()['aux_f1'])

        with open(
                f"{save_dir}/training_dict_cfl_stmt_e{epoch}.json",
                'w') as f:
            json.dump(out_dict, f, indent=2)
        print(json.dumps(out_dict, indent=2))
        print(f1_meter.get())
        if epoch % ConfigClass.save_rate == 0:
            torch.save(
                model.state_dict(),
                f"{save_dir}/training_model_cfl_stmt_e{epoch}.pth")
            edict = eval_func((model, epoch))


def eval(model, dataloader, epoch, save_dir="save"):
    avg_loss, avg_acc = AverageMeter(), AverageMeter()
    f1_meter = BinFullMeter()

    for meter in [avg_loss, avg_acc, f1_meter]:
        meter.reset()

    model.eval()
    bar = tqdm.trange(len(dataloader))
    bar.set_description(f'Eval epoch {epoch}')

    top_1_rec, top_2_rec, top_5_rec, top_10_rec = [
        AverageMeter() for _ in range(4)
    ]
    for i in bar:
        g, stmt_nodes = dataloader[i]
        if g is None:
            continue
        g, stmt_nodes = g.to(device), stmt_nodes.to(device)
        ast_lb = g.lbl[stmt_nodes]
        logits, preds = model(g.xs, g.ess)

        non_zeros_lbs = torch.nonzero(ast_lb).detach()
        if not non_zeros_lbs.shape[0]:
            continue

        ast_lbidxs = torch.flatten(non_zeros_lbs).detach().cpu().tolist()
        loss = F.cross_entropy(logits[stmt_nodes], ast_lb)

        _, ast_cal = torch.max(preds[stmt_nodes].detach().cpu(), dim=1)
        sus_preds = preds[stmt_nodes, 1].detach().cpu()
        top_updates = [(10, top_10_rec), (5, top_5_rec), (2, top_2_rec),
                       (1, top_1_rec)]
        for k, meter in top_updates:
            k = min(stmt_nodes.shape[0], k)
            _, indices = torch.topk(sus_preds, k)
            topk_val = indices[:k].tolist()
            meter.update(int(any([i in ast_lbidxs for i in topk_val])), 1)
        avg_loss.update(loss.item(), stmt_nodes.shape[0])
        avg_acc.update(
            torch.sum(ast_cal.cpu() == ast_lb.cpu()).item() /
            stmt_nodes.shape[0], stmt_nodes.shape[0])
        f1_meter.update(ast_cal.cpu(), ast_lb.cpu())
        bar.set_postfix(ast_loss=loss.item(), acc=avg_acc.avg)

    out_dict = {
        'top_1': top_1_rec.avg,
        'top_2': top_2_rec.avg,
        'top_5': top_5_rec.avg,
        'top_10': top_10_rec.avg,
        'mean_acc': avg_acc.avg,
        'mean_loss': avg_loss.avg,
        'mean_ast_acc': avg_acc.avg,
        'mean_ast_loss': avg_loss.avg,
        'f1': f1_meter.get()['aux_f1']
    }
    with open(
            f"{save_dir}/eval_dict_cfl_stmt_e{epoch}.json", 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(json.dumps(out_dict, indent=2))
    print(f1_meter.get())
    return out_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='codeflaws')
    args = parser.parse_args()
    setattr(args, 'save_dir', f'train_stmt_pyc_cfl_{args.dataset}')
    return args


if __name__ == '__main__':
    args = get_args()
    nx_dataset = CodeflawsCFLNxStatementDataset(
    ) if args.dataset == 'codeflaws' else NBLPyGCFLNxStatementDataset()
    # TODO:
    meta_data = AstGraphMetadata(nx_dataset)
    train_nxs, val_nxs, test_nxs = split_nx_dataset(nx_dataset,
                                                    [0.6, 0.2, 0.2])
    train_pyg_dataset = PyGStatementDataset(
        dataloader=train_nxs,
        meta_data=meta_data,
        ast_enc=None,
        name=f'{args.dataset}_train_pyg_cfl_stmt')
    val_pyg_dataset = PyGStatementDataset(
        dataloader=val_nxs,
        meta_data=meta_data,
        ast_enc=None,
        name=f'{args.dataset}_val_pyg_cfl_stmt')
    test_pyg_dataset = PyGStatementDataset(
        dataloader=test_nxs,
        meta_data=meta_data,
        ast_enc=None,
        name=f'{args.dataset}_test_pyg_cfl_stmt')
    t2id = {'ast': 0, 'test': 1}
    model = MPNNModel_A_T_L(dim_h=64,
                            netypes=len(meta_data.meta_graph),
                            t_srcs=[t2id[e[0]] for e in meta_data.meta_graph],
                            t_tgts=[t2id[e[2]] for e in meta_data.meta_graph],
                            n_al=len(meta_data.t_asts),
                            n_layers=5,
                            n_classes=2).to(device)

    train(model, train_pyg_dataset, 100,
          lambda x: eval(x[0], val_pyg_dataset, x[1], save_dir=args.save_dir),
          0, save_dir=args.save_dir)
    print("Eval test")
    eval(model, test_pyg_dataset, 100)
