from __future__ import print_function, unicode_literals
from utils.train_utils import BinFullMeter, AverageMeter
from utils.utils import ConfigClass
# from utils.drawutils import ast_to_agraph
import pickle as pkl
import json
import os
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from dgl_version.codeflaws.dataloader_cfl_dgl import CodeflawsCFLDGLStatementDataset

from model import GCN_A_L_T_1
import pandas as pd
from utils.common import device


def train(model, dataloader, n_epochs, start_epoch=0):
    opt = torch.optim.Adam(model.parameters())

    mean_ast_loss = AverageMeter()
    mean_ast_acc = AverageMeter()

    tops = {
        t: AverageMeter() for t in ['top_1', 'top_2',
                                    'top_5', 'top_10']}

    f1_rec = BinFullMeter()
    bests = {k: v for k, v in zip(['f1', 'top_1', 'top_2', 'top_5', 'top_10'], [0.0]*6)}
    bests_train = {k: v for k, v in zip(['f1', 'top_1', 'top_2', 'top_5', 'top_10'], [0.0]*6)}

    for epoch in range(n_epochs):
        dataloader.train()
        model.train()
        for m in tops.values()::
            m.reset()

        bar = tqdm.trange(len(dataloader))
        bar.set_description(f'Epoch {epoch}')
        for i in bar:
            g, mask_stmt = dataloader[i]
            if g is None:
                continue

            g, mask_stmt = g.to(device), mask_stmt.to(device)

            ast_lb = g.nodes['ast'].data['tgt'][mask_stmt]
            non_zeros_ast_lbs = torch.nonzero(ast_lb).detach()
            ast_lbidxs = torch.flatten(
                non_zeros_ast_lbs).detach().cpu().tolist()

            g = model(g)

            ast_loss = F.cross_entropy(
                g.nodes['ast'].data['logits'][mask_stmt], ast_lb)

            loss = ast_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            ast_lb = ast_lb.detach().cpu()

            if non_zeros_ast_lbs.shape[0] == 0:
                continue

            _, ast_cal = torch.max(
                g.nodes['ast'].data['logits'][mask_stmt].detach().cpu(),
                dim=1)

            # ALl bugs vs not bugs count
            preds = g.nodes['ast'].data['pred'][mask_stmt, 1].detach().cpu()
            for k, meter in [(k, f'top_{k}') for k in [10, 5, 2, 1]]:
                k = min(stmt_nodes.shape[0], k)
                _, indices = torch.topk(preds, k)
                topk_val = indices[:k].tolist()
                tops[k].update(int(any([i in ast_lbidxs for i in topk_val])), 1)
            mean_ast_loss.update(ast_loss.item(), mask_stmt.shape[0])
            mean_ast_acc.update(
                torch.sum(ast_cal.cpu() == ast_lb.cpu()).item()/mask_stmt.shape[0],
                mask_stmt.shape[0])
            f1_rec.update(ast_cal, ast_lb)
            bar.set_postfix(ast_loss=ast_loss.item(), acc=mean_ast_acc.avg)

        out_dict = {
            'mean_acc': avg_acc.avg, 'mean_loss': avg_loss.avg,
            'mean_ast_acc': avg_acc.avg, 'mean_ast_loss': avg_loss.avg,
            **tops,
            'f1': f1_meter.get()
        }

        for k in ['top_1', 'top_2', 'top_5', 'top_10']:
            bests_train[k] = max(tops[k].avg, bests_train[k].avg)
        if f1_rec.get()['aux_f1'] != 'unk':
            best_f1_train = max(best_f1_train, f1_rec.get()['aux_f1'])

        with open(ConfigClass.trained_dir_codeflaws +
                    f'/training_dict_gumtree_e{epoch}_cfl.json', 'w') as f:
            json.dump(out_dict, f, indent=2)
        print(json.dumps(out_dict, indent=2))
        print(f1_rec.get())

        if epoch % ConfigClass.save_rate == 0:
            eval_dict = eval(model, dataloader, epoch)
            if eval_dict['f1'] != "unk" and eval_dict['f1'] > best_f1:
                best_f1 = eval_dict['f1']
                torch.save(model.state_dict(), os.path.join(
                    ConfigClass.trained_dir_codeflaws, f'model_{epoch}_best_f1_gumtree_cfl.pth'))
        for s in filter(lambda x: eval_dict[x] < bests[x], bests):
            torch.save(
                model.state_dict(), 
                f"{ConfigClass.trained_dir_codeflaws}/model_{epoch}_best_{s}_gumtree_cfl.pth")
                bests[s] = eval_dict[s]
    return {**{f'{s}_train': v for s, v in bests_train},
            **{f'{s}_val': v for s, v in bests}}


def eval(model, dataloader, epoch):
    mean_ast_loss = AverageMeter()
    mean_ast_acc = AverageMeter()

    f1_rec = BinFullMeter()
    dataloader.val()
    mean_ast_loss.reset()
    mean_ast_acc.reset()
    f1_rec.reset()

    model.eval()
    bar = tqdm.trange(len(dataloader))
    bar.set_description(f'Eval epoch {epoch}')

    tops = {t: AverageMeter()
            for t in ['top_1', 'top_2', 'top_5', 'top_10']}

    for i in bar:
        g, mask_stmt = dataloader[i]
        if g is None:
            continue
        g = g.to(device)
        mask_stmt = mask_stmt.to(device)
        ast_lb = g.nodes['ast'].data['tgt'][mask_stmt]

        g = model(g)

        non_zeros_ast_lbs = torch.nonzero(ast_lb).detach()

        ast_lbidxs = torch.flatten(
            non_zeros_ast_lbs).detach().cpu().tolist()

        # Cross entropy is still the same
        ast_loss = F.cross_entropy(
            g.nodes['ast'].data['logits'][mask_stmt], ast_lb)

        if non_zeros_ast_lbs.shape[0] == 0:
            continue

        _, ast_cal = torch.max(
                g.nodes['ast'].data['logits'][mask_stmt].detach().cpu(),
                dim=1)

        # ALl bugs vs not bugs count
        preds = g.nodes['ast'].data['pred'][mask_stmt, 1].detach().cpu()
        for k, meter in [(k, f'top_{k}') for k in [10, 5, 2, 1]]:
            k = min(stmt_nodes.shape[0], k)
            _, indices = torch.topk(preds, k)
            topk_val = indices[:k].tolist()
            tops[k].update(int(any([i in ast_lbidxs for i in topk_val])), 1)

        # mean_loss.update(cfg_loss.item(), g.number_of_nodes('ast'))
        mean_ast_loss.update(ast_loss.item(), mask_stmt.shape[0])
        mean_ast_acc.update(
            torch.sum(ast_cal.cpu() == ast_lb.cpu()).item()/mask_stmt.shape[0],
            mask_stmt.shape[0])

        f1_rec.update(ast_cal, ast_lb)
        bar.set_postfix(ast_loss=ast_loss.item(), acc=mean_ast_acc.avg)

    f1 = f1_rec.get()['aux_f1']
    out_dict = {
        **tops,
        'mean_acc': mean_ast_acc.avg, 'mean_loss': mean_ast_loss.avg,
        'mean_ast_acc': mean_ast_acc.avg,
        'mean_ast_loss': mean_ast_loss.avg,
        'f1': f1
    }

    with open(ConfigClass.result_dir_codeflaws +
              '/eval_dict_e{}_cfl.json'.format(epoch), 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(out_dict)
    return out_dict


if __name__ == '__main__':
    dataset = CodeflawsCFLDGLStatementDataset()
    meta_graph = dataset.meta_graph
    meta_data = CodeflawsCFLStatementGraphMetadata(nx_dataset)
    train_nxs, val_nxs, test_nxs = split_nx_dataset(nx_dataset, [0.6, 0.2, 0.2])
    train_dgl_dataset = CodeflawsCFLDGLStatementDataset(
        dataloader=train_nxs, meta_data=meta_data,
        name='train_dgl_cfl_stmt')
    val_dgl_dataset = CodeflawsCFLDGLStatementDataset(
        dataloader=val_nxs, meta_data=meta_data,
        name='val_dgl_cfl_stmt')
    test_dgl_dataset = CodeflawsCFLSDGLStatementDataset(
        dataloader=test_nxs, meta_data=meta_data,
        name='test_dgl_cfl_stmt')
    model = GCN_A_L_T_1(
        128, meta_graph,
        device=device,
        num_ast_labels=len(dataset.nx_dataset.ast_types),
        num_classes_ast=2).to(device)

    out_dict = train(model, dataset, ConfigClass.n_epochs)
    df = pd.DataFrame(
        columns=[
            'top1_train', 'top1_val', 'top2_train', 'top2_val',
            'top5_train', 'top5_val',
            'top10_train', 'top10_val',
            'f1_train', 'f1_val'
        ])
    df = df.append(out_dict, ignore_index=True)
    df.to_csv(f'out_dict.csv', index=False)
