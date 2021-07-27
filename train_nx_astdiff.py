from __future__ import print_function, unicode_literals
import torch
import os
import torch.nn.functional as F
from dataloader_key_only import CodeflawsFullDGLDataset
from model import HeteroMPNNPredictor1TestNodeType
from utils.utils import ConfigClass
import tqdm
import json
import glob
from utils.train_utils import BinFullMeter, KFullMeter, AverageMeter
import pickle as pkl


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, dataloader, n_epochs):
    opt = torch.optim.Adam(model.parameters())

    # mean_loss = AverageMeter()
    # mean_acc = AverageMeter()
    mean_ast_loss = AverageMeter()
    mean_ast_acc = AverageMeter()

    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()

    f1_meter = KFullMeter(3)
    best_f1 = 0.0
    # model.load_state_dict(torch.load("trained/model_27.pth"))
    # eval(model, dataloader)
    for epoch in range(n_epochs):
        dataloader.train()
        # mean_loss.reset()
        mean_ast_loss.reset()
        mean_ast_acc.reset()
        # mean_acc.reset()
        f1_meter.reset()
        top_10_meter.reset()
        top_5_meter.reset()
        top_2_meter.reset()
        top_1_meter.reset()

        model.train()
        bar = tqdm.trange(len(dataloader))
        bar.set_description(f'Epoch {epoch}')
        for i in bar:
            g = dataloader[i]
            if g is None:
                continue
            # LB will be preprocessed to have
            # lb = g.nodes['cfg'].data['tgt']

            # non_zeros_lbs = torch.nonzero(lb)

            g = g.to(device)
            ast_lb = g.nodes['ast'].data['tgt']
            # lb = lb.to(device)
            g = model(g)

            non_zeros_ast_lbs = torch.nonzero(ast_lb).detach()
            ast_lbidxs = torch.flatten(
                non_zeros_ast_lbs).detach().cpu().tolist()
            # 2 scenario:
            # not using master node
            # logits = g.nodes['cfg'].data['logits']
            # using master node, TODO: To be implemented
            # cfg_loss = F.cross_entropy(logits, lb)
            ast_loss = F.cross_entropy(g.nodes['ast'].data['logits'], ast_lb)
            loss = ast_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ast_lb = ast_lb.detach().cpu()

            if non_zeros_ast_lbs.shape[0] == 0:
                '''
                del g
                torch.cuda.empty_cache()
                '''
                continue
            # loss = cfg_loss + 0.5 * ast_loss

            # _, cal = torch.max(logits, dim=1)
            _, ast_cal = torch.max(
                g.nodes['ast'].data['logits'].detach().cpu(),
                dim=1)

            # ALl bugs vs not bugs count
            preds = - g.nodes['ast'].data['pred'][:, 0].detach().cpu()
            k = min(g.number_of_nodes('ast'), 10)
            _, indices = torch.topk(preds, k)
            top_10_val = indices[:k].tolist()
            top_10_meter.update(
                int(any([idx in ast_lbidxs for idx in top_10_val])), 1)

            k = min(g.number_of_nodes('ast'), 5)
            top_5_val = indices[:k].tolist()
            top_5_meter.update(
                int(any([idx in ast_lbidxs for idx in top_5_val])), 1)

            k = min(g.number_of_nodes('ast'), 2)
            top_2_val = indices[:k].tolist()
            top_2_meter.update(
                int(any([idx in ast_lbidxs for idx in top_2_val])), 1)

            k = min(g.number_of_nodes('ast'), 1)
            top_1_val = indices[:k].tolist()
            top_1_meter.update(int(top_1_val[0] in ast_lbidxs), 1)

            # mean_loss.update(cfg_loss.item(), g.number_of_nodes('ast'))
            mean_ast_loss.update(ast_loss.item(), g.number_of_nodes('ast'))
            '''
            mean_acc.update(
                torch.sum(cal == lb).item()/g.number_of_nodes('ast'),
                g.number_of_nodes('ast'))
            '''
            mean_ast_acc.update(
                torch.sum(ast_cal == ast_lb).item()/g.number_of_nodes('ast'),
                g.number_of_nodes('ast'))
            f1_meter.update(ast_cal, ast_lb)
            '''
            del g
            torch.cuda.empty_cache()
            '''

            bar.set_postfix(ast_loss=ast_loss.item(), acc=mean_ast_acc.avg)

        if epoch % ConfigClass.print_rate == 0:
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
            with open(ConfigClass.result_dir +
                      '/training_dict_e{}.json'.format(epoch), 'w') as f:
                json.dump(out_dict, f, indent=2)
            print(f"loss: {mean_ast_loss.avg}, acc: {mean_ast_acc.avg}, " +
                  f"top 10 acc: {top_10_meter.avg}, " +
                  f"top 5 acc: {top_5_meter.avg}, " +
                  f"top 2 acc {top_2_meter.avg}" +
                  f"top 1 acc {top_1_meter.avg}")
            print(f1_meter.get())
        if epoch % ConfigClass.save_rate == 0:
            l_eval, acc_eval, f1_eval = eval(model, dataloader, epoch)
            if f1_eval != "unk":
                if f1_eval > best_f1:
                    best_f1 = f1_eval
                    torch.save(model.state_dict(), os.path.join(
                        ConfigClass.trained_dir, f'model_{epoch}_best.pth'))


def get_line_mapping(dataloader, real_idx):
    nx_g, _, _, _ = dataloader.nx_dataset[real_idx]
    n_asts = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'ast']
    line = torch.tensor([nx_g.nodes[n]['coord_line'] for n in n_asts],
                        dtype=torch.long)
    return line


def eval_by_line(model, dataloader, epoch, mode='val'):
    # Map from these indices to line
    # Calculate mean scores for these lines
    # Get these unique lines
    # Perform Top K and F1
    if mode == 'val':
        dataloader.val()
    elif mode == 'test':
        dataloader.test()
    f1_meter = BinFullMeter()
    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()
    model.eval()
    out_dict = {}
    line_mapping = None
    if os.path.exists('preprocessed/line_mapping.pkl'):
        line_mapping = pkl.load(open('preprocessed/line_mapping.pkl', 'rb'))
    # Line mapping: index -> ast['line']
    f1_meter.reset()
    top_1_meter.reset()
    top_2_meter.reset()
    top_5_meter.reset()
    top_10_meter.reset()
    line_mapping_changed = False
    for i in tqdm.trange(len(dataloader)):
        real_idx = dataloader.active_idxs[i]
        g = dataloader[i]
        g = g.to(device)
        g = model(g)

        if real_idx not in line_mapping:
            line_mapping[real_idx] = get_line_mapping(dataloader, real_idx)
            line_mapping_changed = True

        g.nodes['ast'].data['line'] = line_mapping[real_idx].to(device)
        all_lines = torch.unique(line_mapping, sorted=True).tolist()
        # Calculate scores by lines
        line_score_tensor = torch.zeros(all_lines.shape[0])
        line_tgt_tensor = torch.zeros(all_lines.shape[0], dtype=torch.long)
        _, g.nodes['ast'].data['new_preds'] = torch.max(
            g.nodes['ast']['preds'], dim=1)

        g.nodes['ast'].data['new_preds'][
            g.nodes['ast'].data['new_preds'] != 0] = 1.0

        line_pred_tensor = torch.zeros((all_lines.shape[0], 2))
        for i, line in enumerate(all_lines):
            mask = g.nodes['ast'].data['line'] == line
            line_score_tensor[i] += torch.sum(
                - g.nodes['ast'].data['preds'][mask][:, 0] + 1.0) /\
                torch.sum(mask)
            line_tgt_tensor[i] += torch.sum(
                g.nodes['ast'].data['tgt'][mask][:, 0])
            line_pred_tensor[i] = torch.sum(
                g.nodes['ast'].data['new_preds'])/torch.sum(mask)

        line_pred_tensor[line_pred_tensor >= 0.5] = 1
        line_pred_tensor[line_pred_tensor < 0.5] = 0

        line_tgt_tensor[line_tgt_tensor > 0] = 1
        non_zeros_lbs = torch.nonzero(line_tgt_tensor)
        if non_zeros_lbs.shape[0] == 0:
            continue
        lbidxs = torch.flatten(non_zeros_lbs).tolist()
        k = min(len(all_lines, 10))
        _, indices = torch.topk(line_score_tensor, k)
        top_10_val = indices[:k].tolist()
        top_10_meter.update(int(any([idx in lbidxs
                                     for idx in top_10_val])), 1)

        k = min(line_score_tensor, 5)
        top_5_val = indices[:k].tolist()
        top_5_meter.update(int(any([idx in lbidxs for idx in top_5_val])), 1)

        k = min(line_score_tensor, 2)
        top_2_val = indices[:k].tolist()
        top_2_meter.update(int(any([idx in lbidxs for idx in top_2_val])), 1)

        k = min(line_score_tensor, 1)
        top_1_val = indices[:k].tolist()
        top_1_meter.update(int(top_1_val[0] in lbidxs), 1)
        f1_meter.update(line_pred_tensor, line_tgt_tensor)

    out_dict['top_1'] = top_1_meter.avg
    out_dict['top_2'] = top_2_meter.avg
    out_dict['top_5'] = top_5_meter.avg
    out_dict['top_10'] = top_10_meter.avg
    out_dict['f1'] = f1_meter.get()
    print(out_dict)
    with open(ConfigClass.result_dir +
              '/eval_dict_by_line_e{}.json'.format(epoch), 'w') as f:
        json.dump(out_dict, f, indent=2)

    if line_mapping_changed:
        pkl.dump(open('preprocessed/line_mapping.pkl', 'wb'))
    return out_dict


def eval(model, dataloader, epoch, mode='val'):
    if mode == 'val':
        dataloader.val()
    elif mode == 'test':
        dataloader.test()
    # mean_loss = AverageMeter()
    # mean_acc = AverageMeter()
    mean_ast_loss = AverageMeter()
    mean_ast_acc = AverageMeter()
    f1_meter = KFullMeter(3)
    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()
    model.eval()
    out_dict = {}
    for i in tqdm.trange(len(dataloader)):
        g = dataloader[i]
        # lb = g.nodes['cfg'].data['tgt']
        if g is None:
            continue

        g = g.to(device)
        # LB will be preprocessed to have
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
        k = min(g.number_of_nodes('ast'), 10)
        _, indices = torch.topk(preds, k)
        top_10_val = indices[:k].tolist()
        top_10_meter.update(
            int(any([idx in ast_lbidxs for idx in top_10_val])), 1)

        k = min(g.number_of_nodes('ast'), 5)
        top_5_val = indices[:k].tolist()
        top_5_meter.update(
            int(any([idx in ast_lbidxs for idx in top_5_val])), 1)

        k = min(g.number_of_nodes('ast'), 2)
        top_2_val = indices[:k].tolist()
        top_2_meter.update(
            int(any([idx in ast_lbidxs for idx in top_2_val])), 1)

        k = min(g.number_of_nodes('ast'), 1)
        top_1_val = indices[:k].tolist()
        top_1_meter.update(int(top_1_val[0] in ast_lbidxs), 1)

        # 2 scenario:
        # not using master node
        # logits = g.nodes['cfg'].data['logits']

        _, ast_cal = torch.max(ast_logits, dim=1)
        mean_ast_loss.update(ast_loss.item(), g.number_of_nodes('ast'))
        mean_ast_acc.update(
            torch.sum(ast_cal == ast_lb).item()/g.number_of_nodes('ast'),
            g.number_of_nodes('ast'))
        f1_meter.update(ast_cal, ast_lb)
    out_dict['top_1'] = top_1_meter.avg
    out_dict['top_2'] = top_2_meter.avg
    out_dict['top_5'] = top_5_meter.avg
    out_dict['top_10'] = top_10_meter.avg
    out_dict['mean_acc'] = mean_ast_acc.avg
    out_dict['mean_loss'] = mean_ast_loss.avg
    out_dict['f1'] = f1_meter.get()
    with open(ConfigClass.result_dir + f'/eval_dict_e{epoch}.json', 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(out_dict)
    return mean_ast_loss.avg, mean_ast_acc.avg, f1_meter.get()['aux_f1']


if __name__ == '__main__':
    # config
    dataset_opt = 'codeflaws'  # nbl, codeflaws
    graph_opt = 2  # 1, 2
    # loaddataset
    dataset = CodeflawsFullDGLDataset()
    meta_graph = dataset.meta_graph
    model = HeteroMPNNPredictor1TestNodeType(
        len(ConfigClass.cfg_label_corpus),
        dataset.cfg_content_dim,
        128, 32, meta_graph, num_classes=2,
        device=device, num_ast_labels=len(dataset.nx_dataset.ast_types),
        ast_content_feats=dataset.ast_content_dim, num_classes_ast=3)
    ConfigClass.preprocess_dir = "{}/{}/{}".format(
        ConfigClass.preprocess_dir, dataset_opt, graph_opt)
    # train(model, dataset, ConfigClass.n_epochs)
    list_models_paths = list(
        glob.glob(f"{ConfigClass.trained_dir}/model*best.pth"))
    for model_path in list_models_paths:
        epoch = int(model_path.split("_")[1])
        print(f"Evaluating {model_path}:")
        model.load_state_dict(torch.load(model_path))
        print("val: ")
        eval_by_line(model, dataset, epoch, 'val')
        print('test: ')
        eval_by_line(model, dataset, epoch, 'test')
    best_latest = max(int(model_path.split("_")[1])
                      for model_path in list_models_paths)
    model_path = f"{ConfigClass.trained_dir}/model_{best_latest}_best.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Evaluation: {model_path}")
    dataset.test()
    eval(model, dataset, best_latest)
