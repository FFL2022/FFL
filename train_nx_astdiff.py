from __future__ import print_function, unicode_literals
import math
import time
import torch
import os
import torch.nn.functional as F
from dataloader_key_only import CodeflawsFullDGLDataset
from model import HeteroMPNNPredictor1TestNodeType
from utils.utils import ConfigClass
import tqdm
import json
import glob


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AverageMeter(object):
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count


class BinFullMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tn = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, cal, labels):
        self.tp += torch.sum((cal[labels == 1] == 1)).item()
        self.tn += torch.sum((cal[labels == 0] == 0)).item()
        self.fp += torch.sum((cal[labels == 0] == 1)).item()
        self.fn += torch.sum((cal[labels == 1] == 0)).item()

    def get(self):
        tnr = 'unk'
        tpr = 'unk'
        prec = 'unk'
        rec = 'unk'
        aux_f1 = 'unk'
        if self.tn + self.tp > 0:
            tnr = self.tn/(self.tn + self.tp)
        if (self.tp + self.tn) > 0:
            tpr = self.tp/(self.tp + self.fn)
        if (self.tp + self.fp) > 0:
            prec = self.tp/(self.tp + self.fp)
        if (self.tp + self.fn) > 0:
            rec = self.tp/(self.tp + self.fn)
        if prec != 'unk' and rec != 'unk':
            aux_f1 = (prec + rec)/2
        return {'tpr': tpr, 'tnr': tnr, 'prec': prec, 'rec': rec,
                'aux_f1': aux_f1}


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))


def train(model, dataloader, n_epochs):
    opt = torch.optim.Adam(model.parameters())

    mean_loss = AverageMeter()
    mean_ast_loss = AverageMeter()
    mean_acc = AverageMeter()
    mean_ast_acc = AverageMeter()

    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()

    f1_meter = BinFullMeter()
    best_f1 = 0.0
    # model.load_state_dict(torch.load("trained/model_27.pth"))
    # eval(model, dataloader)
    for epoch in range(n_epochs):
        dataloader.train()
        mean_loss.reset()
        mean_ast_loss.reset()
        mean_ast_acc.reset()
        mean_acc.reset()
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
            lb = g.nodes['cfg'].data['tgt']
            ast_lb = g.nodes['cfg'].data['tgt']

            non_zeros_lbs = torch.nonzero(lb)
            if non_zeros_lbs.shape[0] == 0:
                continue
            g = g.to(device)
            lbidxs = torch.flatten(non_zeros_lbs).tolist()
            lb = lb.to(device)
            ast_lb = ast_lb.to(device)
            g = model(g)
            # 2 scenario:
            # not using master node
            logits = g.nodes['cfg'].data['logits']
            # using master node, TODO: To be implemented
            cfg_loss = F.cross_entropy(logits, lb)
            ast_loss = F.cross_entropy(g.nodes['ast'].data['logits'])
            loss = cfg_loss + 0.5 * ast_loss

            _, cal = torch.max(logits, dim=1)
            _, ast_cal = torch.max(g.nodes['ast'].data['logits'], dim=1)

            preds = g.nodes['cfg'].data['pred'][:, 1]
            k = min(g.number_of_nodes('cfg'), 10)
            _, indices = torch.topk(preds, k)
            top_10_val = indices[:k].tolist()
            top_10_meter.update(
                int(any([idx in lbidxs for idx in top_10_val])), 1)

            k = min(g.number_of_nodes('cfg'), 5)
            top_5_val = indices[:k].tolist()
            top_5_meter.update(
                int(any([idx in lbidxs for idx in top_5_val])), 1)

            k = min(g.number_of_nodes('cfg'), 2)
            top_2_val = indices[:k].tolist()
            top_2_meter.update(
                int(any([idx in lbidxs for idx in top_2_val])), 1)

            k = min(g.number_of_nodes('cfg'), 1)
            top_1_val = indices[:k].tolist()
            top_1_meter.update(int(top_1_val[0] in lbidxs), 1)

            mean_loss.update(cfg_loss.item(), g.number_of_nodes('cfg'))
            mean_ast_loss.update(ast_loss.item(), g.number_of_nodes('ast'))
            mean_acc.update(
                torch.sum(cal == lb).item()/g.number_of_nodes('cfg'),
                g.number_of_nodes('cfg'))
            mean_ast_acc.update(
                torch.sum(ast_cal == ast_lb).item()/g.number_of_nodes('ast'),
                g.number_of_nodes('ast'))
            f1_meter.update(cal, lb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bar.set_postfix(loss=loss.item(), acc=mean_acc.avg,
                            ast_loss=ast_loss.item())

        if epoch % ConfigClass.print_rate == 0:
            out_dict = {}
            out_dict['top_1'] = top_1_meter.avg
            out_dict['top_2'] = top_2_meter.avg
            out_dict['top_5'] = top_5_meter.avg
            out_dict['top_10'] = top_10_meter.avg
            out_dict['mean_acc'] = mean_acc.avg
            out_dict['mean_loss'] = mean_loss.avg
            out_dict['mean_ast_acc'] = mean_ast_acc.avg
            out_dict['mean_ast_loss'] = mean_ast_loss.avg
            out_dict['f1'] = f1_meter.get()
            with open(ConfigClass.result_dir +
                      '/training_dict_e{}.json'.format(epoch), 'w') as f:
                json.dump(out_dict, f, indent=2)
            print(f"loss: {mean_loss.avg}, acc: {mean_acc.avg}, " +
                  f"top 10 acc: {top_10_meter.avg}, " +
                  f"top 5 acc: {top_5_meter.avg}, top 2 acc {top_2_meter.avg}")
            print(f1_meter.get())
        if epoch % ConfigClass.save_rate == 0:
            l_eval, acc_eval, f1_eval = eval(model, dataloader, epoch)
            if f1_eval != "unk":
                if f1_eval > best_f1:
                    best_f1 = f1_eval
                    torch.save(model.state_dict(), os.path.join(
                        ConfigClass.trained_dir, f'model_{epoch}_best.pth'))


def eval(model, dataloader, epoch):
    dataloader.val()
    mean_loss = AverageMeter()
    mean_acc = AverageMeter()
    f1_meter = BinFullMeter()
    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()
    model.eval()
    out_dict = {}
    for i in tqdm.trange(len(dataloader)):
        g = dataloader[i]
        lb = g.nodes['cfg'].data['tgt']
        if g is None or lb is None:
            continue

        non_zeros_lbs = torch.nonzero(lb)
        if non_zeros_lbs.shape[0] == 0:
            continue

        g = g.to(device)
        # LB will be preprocessed to have
        lb = lb.to(device)
        model(g)
        lbidxs = torch.flatten(non_zeros_lbs).tolist()

        preds = g.nodes['cfg'].data['pred'][:, 1]
        k = min(g.number_of_nodes('cfg'), 10)
        _, indices = torch.topk(preds, k)
        top_10_val = indices[:k].tolist()
        top_10_meter.update(int(any([idx in lbidxs for idx in top_10_val])), 1)

        k = min(g.number_of_nodes('cfg'), 5)
        top_5_val = indices[:k].tolist()
        top_5_meter.update(int(any([idx in lbidxs for idx in top_5_val])), 1)

        k = min(g.number_of_nodes('cfg'), 2)
        top_2_val = indices[:k].tolist()
        top_2_meter.update(int(any([idx in lbidxs for idx in top_2_val])), 1)

        k = min(g.number_of_nodes('cfg'), 1)
        top_1_val = indices[:k].tolist()
        top_1_meter.update(int(top_1_val[0] in lbidxs), 1)

        # 2 scenario:
        # not using master node
        logits = g.nodes['cfg'].data['logits']
        # using master node, to be implemented
        loss = F.cross_entropy(logits, lb)
        _, cal = torch.max(logits, dim=1)
        mean_loss.update(loss.item(), g.number_of_nodes('cfg'))
        mean_acc.update(torch.sum(cal == lb).item()/g.number_of_nodes('cfg'),
                        g.number_of_nodes('cfg'))
        f1_meter.update(cal, lb)
    out_dict['top_1'] = top_1_meter.avg
    out_dict['top_2'] = top_2_meter.avg
    out_dict['top_5'] = top_5_meter.avg
    out_dict['top_10'] = top_10_meter.avg
    out_dict['mean_acc'] = mean_acc.avg
    out_dict['mean_loss'] = mean_loss.avg
    out_dict['f1'] = f1_meter.get()
    with open(ConfigClass.result_dir + f'/eval_dict_e{epoch}.json', 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(out_dict)
    return mean_loss.avg, mean_acc.avg, f1_meter.get()['aux_f1']


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
        256, 32, meta_graph, 2, device,
        len(dataset.nx_dataset.ast_types),
        dataset.ast_content_dim, 3)
    ConfigClass.preprocess_dir = "{}/{}/{}".format(
        ConfigClass.preprocess_dir, dataset_opt, graph_opt)
    train(model, dataset, ConfigClass.n_epochs)
    list_models_paths = list(
        glob.glob(f"{ConfigClass.trained_dir}/model*best.pth"))
    best_latest = max(int(model_path.split("_")[1])
                      for model_path in list_models_paths)
    model_path = f"{ConfigClass.trained_dir}/model_{best_latest}_best.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Evaluation: {model_path}")
    dataset.test()
    eval(model, dataset, best_latest)
