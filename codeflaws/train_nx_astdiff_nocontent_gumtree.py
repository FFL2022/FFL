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
from codeflaws.dataloader_gumtree import CodeflawsGumtreeDGLStatementDataset

from model import GCN_A_L_T_1
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, dataloader, n_epochs, start_epoch=0):
    opt = torch.optim.Adam(model.parameters())

    mean_ast_loss = AverageMeter()
    mean_ast_acc = AverageMeter()

    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()

    f1_meter = BinFullMeter()
    best_f1 = 0.0
    best_top1 = 0.0
    best_top2 = 0.0
    best_top5 = 0.0
    best_top10 = 0.0

    best_top1_train = 0.0
    best_top2_train = 0.0
    best_top5_train = 0.0
    best_top10_train = 0.0
    best_f1_train = 0.0

    for epoch in range(n_epochs):
        dataloader.train()
        model.train()

        mean_ast_loss.reset()
        mean_ast_acc.reset()
        f1_meter.reset()
        top_10_meter.reset()
        top_5_meter.reset()
        top_2_meter.reset()
        top_1_meter.reset()

        bar = tqdm.trange(len(dataloader))
        bar.set_description(f'Epoch {epoch}')
        for i in bar:
            g, mask_stmt = dataloader[i]
            if g is None:
                continue

            g = g.to(device)
            mask_stmt = mask_stmt.to(device)

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
            k = min(mask_stmt.shape[0], 10)
            _, indices = torch.topk(preds, k)
            top_10_val = indices[:k].tolist()
            top_10_meter.update(
                int(any([idx in ast_lbidxs for idx in top_10_val])), 1)

            k = min(mask_stmt.shape[0], 5)
            top_5_val = indices[:k].tolist()
            top_5_meter.update(
                int(any([idx in ast_lbidxs for idx in top_5_val])), 1)

            k = min(mask_stmt.shape[0], 2)
            top_2_val = indices[:k].tolist()
            top_2_meter.update(
                int(any([idx in ast_lbidxs for idx in top_2_val])), 1)

            k = min(mask_stmt.shape[0], 1)
            top_1_val = indices[:k].tolist()
            top_1_meter.update(int(top_1_val[0] in ast_lbidxs), 1)

            # mean_loss.update(cfg_loss.item(), g.number_of_nodes('ast'))
            mean_ast_loss.update(ast_loss.item(), mask_stmt.shape[0])
            mean_ast_acc.update(
                torch.sum(ast_cal == ast_lb).item()/g.number_of_nodes('ast'),
                g.number_of_nodes('ast'))
            f1_meter.update(ast_cal, ast_lb)
            bar.set_postfix(ast_loss=ast_loss.item(), acc=mean_ast_acc.avg)

        out_dict = {}
        out_dict['top_1'] = top_1_meter.avg
        out_dict['top_2'] = top_2_meter.avg
        out_dict['top_5'] = top_5_meter.avg
        out_dict['top_10'] = top_10_meter.avg
        out_dict['mean_acc'] = mean_ast_acc.avg
        out_dict['mean_loss'] = mean_ast_loss.avg
        out_dict['mean_ast_acc'] = mean_ast_acc.avg
        out_dict['mean_ast_loss'] = mean_ast_loss.avg
        out_dict['f1'] = f1_meter.get()

        best_top1_train = max(top_1_meter.avg, best_top1_train)
        best_top2_train = max(top_2_meter.avg, best_top2_train)
        best_top5_train = max(top_5_meter.avg, best_top5_train)
        best_top10_train = max(top_10_meter.avg, best_top10_train)
        if f1_meter.get()['aux_f1'] != 'unk':
            best_f1_train = max(best_f1_train, f1_meter.get()['aux_f1'])

        with open(Config.result_dir_d4j +
                    f'/training_dict_e{epoch}.json', 'w') as f:
            json.dump(out_dict, f, indent=2)
        print(f"loss: {mean_ast_loss.avg}, acc: {mean_ast_acc.avg}, " +
                f"top 10 acc: {top_10_meter.avg}, " +
                f"top 5 acc: {top_5_meter.avg}, " +
                f"top 2 acc {top_2_meter.avg}, " +
                f"top 1 acc {top_1_meter.avg}, ")
        print(f1_meter.get())

        if epoch % Config.save_rate == 0:
            eval_dict = eval(model, dataloader, epoch)
            if eval_dict['f1'] != "unk":
                if eval_dict['f1'] > best_f1:
                    best_f1 = eval_dict['f1']
                    torch.save(model.state_dict(), os.path.join(
                        Config.trained_dir_d4j, f'model_{epoch}_best_f1.pth'))
            if eval_dict['top_1'] > best_top1:
                torch.save(model.state_dict(), os.path.join(
                    Config.trained_dir_d4j, f'model_{epoch}_best_top1.pth'))
                best_top1 = eval_dict['top_1']
            if eval_dict['top_2'] > best_top2:
                torch.save(model.state_dict(), os.path.join(
                    Config.trained_dir_d4j, f'model_{epoch}_best_top2.pth'))
                best_top2 = eval_dict['top_2']
            if eval_dict['top_5'] > best_top5:
                torch.save(model.state_dict(), os.path.join(
                    Config.trained_dir_d4j, f'model_{epoch}_best_top5.pth'))
                best_top5 = eval_dict['top_5']
            if eval_dict['top_10'] > best_top10:
                torch.save(model.state_dict(), os.path.join(
                    Config.trained_dir_d4j, f'model_{epoch}_best_top10.pth'))
                best_top10 = eval_dict['top_10']
    return {'top1_train': best_top1_train,
            'top1_val': best_top1,
            'top2_train': best_top2_train,
            'top2_val': best_top2,
            'top5_train': best_top5_train,
            'top5_val': best_top5,
            'top10_train': best_top10_train,
            'top10_val': best_top10,
            'f1_train': best_f1_train,
            'f1_val': best_f1}


def eval(model, dataloader, epoch):
    mean_ast_loss = AverageMeter()
    mean_ast_acc = AverageMeter()

    f1_meter = BinFullMeter()
    dataloader.val()
    mean_ast_loss.reset()
    mean_ast_acc.reset()
    f1_meter.reset()

    model.eval()
    bar = tqdm.trange(len(dataloader))
    bar.set_description(f'Eval epoch {epoch}')
    best_top1 = 0
    best_top2 = 0
    best_top5 = 0
    best_top10 = 0
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

        ast_lb = ast_lb.detach().cpu()
        # ALl bugs vs not bugs count
        # But recall are filtered
        preds = (
            g.nodes['ast'].data['pred'][mask_stmt, 1] *
            mask[mask_stmt]
        ).detach().cpu()



        k = min(mask_stmt.shape[0], 10)
        _, indices = torch.topk(preds, k)
        top_10_val = indices[:k].tolist()
        best_top10 = max(
            int(any([idx in ast_lbidxs for idx in top_10_val])), best_top10)

        k = min(mask_stmt.shape[0], 5)
        top_5_val = indices[:k].tolist()
        best_top5 = max(
            int(any([idx in ast_lbidxs for idx in top_5_val])), best_top5)

        k = min(mask_stmt.shape[0], 2)
        top_2_val = indices[:k].tolist()
        best_top2 = max(int(any([idx in ast_lbidxs for idx in top_2_val])),
                        best_top2)

        k = min(mask_stmt.shape[0], 1)
        top_1_val = indices[:k].tolist()
        best_top1 = max(int(top_1_val[0] in ast_lbidxs), best_top1)

        # mean_loss.update(cfg_loss.item(), g.number_of_nodes('ast'))
        mean_ast_loss.update(ast_loss.item(), mask_stmt.shape[0])
        mean_ast_acc.update(
            torch.sum(ast_cal == ast_lb).item()/g.number_of_nodes('ast'),
            g.number_of_nodes('ast'))
        f1_meter.update(ast_cal, ast_lb)
        bar.set_postfix(ast_loss=ast_loss.item(), acc=mean_ast_acc.avg)

    out_dict = {}
    out_dict['top_1'] = best_top1
    out_dict['top_2'] = best_top2
    out_dict['top_5'] = best_top5
    out_dict['top_10'] = best_top10
    out_dict['mean_acc'] = mean_ast_acc.avg
    out_dict['mean_loss'] = mean_ast_loss.avg
    out_dict['mean_ast_acc'] = mean_ast_acc.avg
    out_dict['mean_ast_loss'] = mean_ast_loss.avg
    f1 = f1_meter.get()['aux_f1']

    out_dict['f1'] = f1
    with open(Config.result_dir_d4j +
              '/eval_dict_e{}.json'.format(epoch), 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(f"loss: {mean_ast_loss.avg}, acc: {mean_ast_acc.avg}, " +
          f"top 10 acc: {best_top10}, " +
          f"top 5 acc: {best_top5}, " +
          f"top 2 acc: {best_top2}, " +
          f"top 1 acc: {best_top1}, " +
          f'f1: {f1}'
          )
    return out_dict



if __name__ == '__main__':
    dataset = CodeflawsGumtreeDGLStatementDataset()
    meta_graph = dataset.meta_graph

    model = GCN_A_L_T_1(
        128, meta_graph,
        device=device, 
        num_ast_labels=len(dataset.nx_dataset.ast_types),
        num_classes_ast=2)

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