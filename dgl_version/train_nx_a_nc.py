from __future__ import print_function, unicode_literals
import torch
import os
import torch.nn.functional as F
from nbl.dataloader_key_only import NBLNxDataset
from codeflaws.dataloader_key_only import CodeflawsNxDataset
from utils.data_utils import NxDataloader
from dgl_version.dataloader_key_only import ASTDGLDataset
from dgl_version.model import GCN_A_L_T_1
from utils.utils import ConfigClass
from utils.draw_utils import ast_to_agraph
from utils.data_utils import AstGraphMetadata
import tqdm
import json
import glob
from utils.train_utils import BinFullMeter, KFullMeter, AverageMeter
from graph_algos.nx_shortcuts import nodes_where
import pickle as pkl
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_dir = ConfigClass.trained_dir_nbl
result_dir = ConfigClass.result_dir_nbl


def train(model, dataloader, val_dataloader, n_epochs, start_epoch=0):
    opt = torch.optim.Adam(model.parameters())

    mean_ast_loss = AverageMeter()
    mean_ast_acc = AverageMeter()

    tops = {f'top_{k}': AverageMeter for k in [1, 3, 5, 10]}

    f1_meter = KFullMeter(3)
    bests = {'f1': 0.0, **{t: -1.0 for t in tops}}
    for epoch in range(n_epochs):
        dataloader.train()
        for m in list(tops.values()) + [
                mean_ast_loss, mean_ast_acc, f1_meter]:
            m.reset()

        model.train()
        bar = tqdm.trange(len(dataloader))
        #bar = tqdm.trange(5)
        bar.set_description(f'Epoch {epoch}')
        for i in bar:

            g = dataloader[i]
            if g is None:
                continue
            g = g.to(device)
            ast_lb = g.nodes['ast'].data['tgt']
            g = model(g)

            non_zeros_ast_lbs = torch.nonzero(ast_lb).detach()
            ast_lbidxs = torch.flatten(
                non_zeros_ast_lbs).detach().cpu().tolist()
            
            ast_loss = F.cross_entropy(g.nodes['ast'].data['logits'], ast_lb)
            loss = ast_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ast_lb = ast_lb.detach().cpu()

            if non_zeros_ast_lbs.shape[0] == 0:
                continue
            _, ast_cal = torch.max(
                g.nodes['ast'].data['logits'].detach().cpu(),
                dim=1)

            # ALl bugs vs not bugs count
            preds = - g.nodes['ast'].data['pred'][:, 0].detach().cpu()
            for k in [10, 5, 2, 1]:
                k = min(g.number_of_nodes('ast'), k)
                _, indices = torch.topk(preds, k)
                top_k_val = indices[:k].tolist()
                tops[f'top_{k}'].update(
                    int(any([idx in ast_lbidxs for idx in top_k_val])), 1)

            # mean_loss.update(cfg_loss.item(), g.number_of_nodes('ast'))
            mean_ast_loss.update(ast_loss.item(), g.number_of_nodes('ast'))
            mean_ast_acc.update(
                torch.sum(ast_cal == ast_lb).item()/g.number_of_nodes('ast'),
                g.number_of_nodes('ast'))
            f1_meter.update(ast_cal, ast_lb)
            
            bar.set_postfix(ast_loss=ast_loss.item(), acc=mean_ast_acc.avg)

        if epoch % ConfigClass.print_rate == 0:
            out_dict = {
                **tops, 'mean_acc': mean_ast_acc.avg
                'mean_loss': mean_ast_loss.avg,
                'mean_ast_loss': mean_ast_loss.avg,
                'mean_ast_acc': mean_ast_acc.avg,
                'f1': f1_meter.get()
            }
            with open(
                    f'{result_dir}/training_dict_e{epoch}.json', 'w') as f:
                json.dump(out_dict, f, indent=2)
            out_dict
        if epoch % ConfigClass.save_rate == 0:
            eval_dict  = eval_by_line(model, val_dataloader, epoch)
            if eval_dict['f1']['aux_f1'] != "unk":
                if eval_dict['f1']['aux_f1'] > best_f1:
                    best_f1 = eval_dict['f1']['aux_f1']
                    torch.save(
                        model.state_dict(), os.path.join(
                        f"{trained_dir}/model_{epoch}_best_f1.pth"))
            for t in tops:
                if eval_dict[t] > bests[t]:
                    bests[t] = eval_dict[t]
                    torch.save(model.state_dict(),
                            f"{trained_dir}/model_{epoch}_best_{t}.pth")
        print("Best_result: ", bests)
        torch.save(model.state_dict(), f"{trained_dir}/model_last.pth")


def get_line_mapping(dataloader, real_idx):
    nx_g, _, _, _ = dataloader.nx_dataset[real_idx]
    n_asts = nodes_where(nx_g, graph='ast')
    line = torch.tensor([nx_g.nodes[n]['start_line'] for n in n_asts],
                        dtype=torch.long)
    return line


def map_from_predict_to_node(dataloader, real_idx, node_preds, tgts):
    nx_g, _, _, _ = dataloader.nx_dataset[real_idx]
    n_asts = nodes_where(nx_g, graph='ast')
    for i, n in enumerate(n_asts):
        nx_g.nodes[n]['status'] = 0
        if node_preds[i] == 0:
            continue
        if tgts[i] == 0:
            nx_g.nodes[n]['status'] = 6 + node_preds[i]
        elif tgts[i] == 1:
            if node_preds[i] == tgts[i]:
                nx_g.nodes[n]['status'] = 3
            else:
                nx_g.nodes[n]['status'] = 5
        elif tgts[i] == 2:
            if node_preds[i] == tgts[i]:
                nx_g.nodes[n]['status'] = 4
            else:
                nx_g.nodes[n]['status'] = 6
    return nx_g.subgraph(n_asts)


def eval_by_line(model, dataloader, epoch=-1, mode='val', draw = False):
    if mode == 'val':
        dataloader.val()

    os.makedirs(f'images_nbl_{epoch}', exist_ok=True)
    f1_meter = BinFullMeter()
    tops = {f'top_{k}': AverageMeter() for k in [1, 3, 5, 10]}
    model.eval()
    out_dict = {}
    line_mapping = {}
    if os.path.exists('preprocessed/nbl/line_mapping.pkl'):
        line_mapping = pkl.load(open('preprocessed/nbl/line_mapping.pkl', 'rb'))
    # Line mapping: index -> ast['line']
    f1_meter.reset()
    for m in tops.values:
        m.reset()
    line_mapping_changed = False
    for i in tqdm.trange(len(dataloader)):
    #for i in tqdm.trange(1):
        real_idx = dataloader.active_idxs[i]
        g = dataloader[i]
        g = g.to(device)
        g = model(g)

        if real_idx not in line_mapping:
            line_mapping[real_idx] = get_line_mapping(dataloader, real_idx)
            line_mapping_changed = True

        g.nodes['ast'].data['line'] = line_mapping[real_idx].to(device)
        all_lines = torch.unique(line_mapping[real_idx], sorted=True).tolist()
        print(len(all_lines))
        # Calculate scores by lines
        line_score_tensor = torch.zeros(len(all_lines)).to(device)
        line_tgt_tensor = torch.zeros(
            len(all_lines), dtype=torch.long).to(device)
        _, g.nodes['ast'].data['new_pred'] = torch.max(
            g.nodes['ast'].data['pred'], dim=1)

        nx_g = map_from_predict_to_node(
            dataloader, real_idx,
            g.nodes['ast'].data['new_pred'].detach().cpu().numpy(),
            g.nodes['ast'].data['tgt'].detach().cpu().numpy()
        )

        if nx_g.number_of_nodes() > 1000 and not draw:
            continue
        try:
            ast_to_agraph(nx_g, f'images_nbl_{epoch}/{real_idx}.png',
                          take_content=False)
        except:
            continue

        g.nodes['ast'].data['new_pred'][
            g.nodes['ast'].data['new_pred'] != 0] = 1.0

        line_pred_tensor = torch.zeros(len(all_lines))
        for i, line in enumerate(all_lines):
            mask = (g.nodes['ast'].data['line'] == line).to(device)
            # Max, Mean
            line_score_tensor[i] += torch.sum(
                - g.nodes['ast'].data['pred'][mask][:, 0] + 1.0) /\
                torch.sum(mask)
            line_tgt_tensor[i] += torch.sum(
                g.nodes['ast'].data['tgt'][mask])
            line_pred_tensor[i] = torch.sum(
                g.nodes['ast'].data['new_pred'])/torch.sum(mask)

        line_pred_tensor[line_pred_tensor >= 0.5] = 1
        line_pred_tensor[line_pred_tensor < 0.5] = 0

        line_tgt_tensor[line_tgt_tensor > 0] = 1
        non_zeros_lbs = torch.nonzero(line_tgt_tensor)
        if non_zeros_lbs.shape[0] == 0:
            continue
        lbidxs = torch.flatten(non_zeros_lbs).tolist()
        for k in [10, 5, 2, 1]:
            real_k = min(len(all_lines), 10)
            _, indices = torch.topk(line_score_tensor, real_k)
            top_k_val = indices[:real_k].tolist()
            tops[f'top_{k}'].update(int(any([idx in lbidxs
                for idx in top_k_val])), 1)

        f1_meter.update(line_pred_tensor, line_tgt_tensor)

    out_dict = {**tops, 'f1': f1_meter.get()}
    print(out_dict)
    with open(f"{trained_dir}/eval_dict_by_line_e{epoch}.json", 'w') as f:
        json.dump(out_dict, f, indent=2)

    if line_mapping_changed:
        pkl.dump(line_mapping, open('preprocessed/nbl/line_mapping.pkl', 'wb'))
    return out_dict


def eval(model, dataloader, epoch=-1, mode='val'):
    if mode == 'val':
        dataloader.val()
    mean_ast_loss = AverageMeter()
    mean_ast_acc = AverageMeter()
    f1_meter = KFullMeter(3)
    tops = {f'top_{k}': AverageMeter() for k in [1, 3, 5, 10]}
    model.eval()
    out_dict = {}
    for i in tqdm.trange(len(dataloader)):
        g = dataloader[i]
        if g is None:
            continue

        g = g.to(device)
        ast_lb = g.nodes['ast'].data['tgt']
        ast_non_zeros_lbs = torch.nonzero(ast_lb).cpu()
        model(g)

        if ast_non_zeros_lbs.shape[0] == 0:
            continue
        ast_lbidxs = torch.flatten(ast_non_zeros_lbs).tolist()
        ast_logits = g.nodes['ast'].data['logits']

        # using master node, to be implemented
        ast_loss = F.cross_entropy(g.nodes['ast'].data['logits'], ast_lb)
        # cfg_loss = F.cross_entropy(logits, lb)
        preds = - g.nodes['ast'].data['pred'][:, 0]
        for k in [10, 5, 2, 1]:
            real_k = min(g.number_of_nodes('ast'), k)
            _, indices = torch.topk(preds, real_k)
            top_k_val = indices[:real_k].tolist()
            tops[f'top_{k}'].update(
                int(any([idx in ast_lbidxs for idx in top_k_val])), 1)


        _, ast_cal = torch.max(ast_logits, dim=1)
        mean_ast_loss.update(ast_loss.item(), g.number_of_nodes('ast'))
        mean_ast_acc.update(
            torch.sum(ast_cal == ast_lb).item()/g.number_of_nodes('ast'),
            g.number_of_nodes('ast'))
        f1_meter.update(ast_cal, ast_lb)
    out_dict = {
            **tops, 'mean_acc': mean_ast_acc.avg,
            'mean_loss': mean_ast_loss.avg, 'f1': f1_meter.get()}
    with open(f"{result_dir}/eval_dict_e{epoch}.json", 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(out_dict)
    return mean_ast_loss.avg, mean_ast_acc.avg, f1_meter.get()['aux_f1']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nbl')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    NxDatasetClass = NBLNxDataset if args.dataset == 'nbl' else CodeflawsNxDataset
    dataset = NxDatasetClass()
    meta_data = AstGraphMetadata(dataset)
    meta_graph = dataset.meta_graph
    trained_dir = ConfigClass.trained_dir_nbl if args.dataset == 'nbl' else ConfigClass.trained_dir_codeflaws
    result_dir = ConfigClass.result_dir_nbl if args.dataset == 'nbl' else ConfigClass.result_dir_codeflaws
    train_dgl_dataset = ASTDGLDataset(
        dataloader=train_nxs, meta_data=meta_data,
        name='train_{args.dataset}_cfl_ast_node',
        save_dir=result_dir)
    val_dgl_dataset = ASTDGLDataset(
        dataloader=val_nxs, meta_data=meta_data,
        name='val_{args.dataset}_cfl_ast_node',
        save_dir=result_dir)
    test_dgl_dataset = ASTDGLDataset(
        dataloader=test_nxs, meta_data=meta_data,
        name='test_{args.dataset}_cfl_ast_node',
        save_dir=result_dir)

    model = GCN_A_L_T_1(
        128, meta_graph,
        device=device, num_ast_labels=len(dataset.nx_dataset.ast_types),
        num_classes_ast=3).to(device)

    train(model, train_dgl_dataset, 50, 0)
    dataset.val()
    eval_by_line(model, test_dgl_dataset)
