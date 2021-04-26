from __future__ import print_function, unicode_literals

import math
import time
import torch
import os
import torch.nn.functional as F
from dataloader import default_dataset
from model import HeteroMPNNPredictor
from utils.utils import ConfigClass
import tqdm
import json


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
    mean_acc = AverageMeter()

    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()

    f1_meter = BinFullMeter()
    best_f1 = 0.0
    for epoch in range(n_epochs):
        dataloader.train()
        mean_loss.reset()
        mean_acc.reset()
        f1_meter.reset()
        top_10_meter.reset()
        top_5_meter.reset()
        top_2_meter.reset()
        top_1_meter.reset()

        model.train()

        for i in tqdm.trange(len(dataloader)):
            g, lb = dataloader[i]
            if g is None or lb is None:
                continue
            # LB will be preprocessed to have
            non_zeros_lbs = torch.nonzero(lb)
            if non_zeros_lbs.shape[0] == 0:
                continue
            g = g.to(device)
            lbidxs = torch.flatten(non_zeros_lbs).tolist()
            lb = lb.to(device)
            model(g)
            # 2 scenario:
            # not using master node
            logits = g.nodes['cfg'].data['logits']
            # using master node, to be implemented
            loss = F.cross_entropy(logits, lb)

            _, cal = torch.max(logits, dim=1)


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

            mean_loss.update(loss.item(), g.number_of_nodes('cfg'))
            mean_acc.update(torch.sum(cal == lb).item()/g.number_of_nodes('cfg'),
                            g.number_of_nodes('cfg'))
            f1_meter.update(cal, lb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % ConfigClass.print_rate == 0:
            print("loss: {}, acc: {}, top 10 acc: {}, top 5 acc: {}, top 2 acc {}".format(mean_loss.avg, mean_acc.avg,
                top_10_meter.avg, top_5_meter.avg, top_2_meter.avg))
            print(f1_meter.get())
        if epoch % ConfigClass.save_rate == 0:
            l_eval, acc_eval, f1_eval = eval(model, dataloader)
            if f1_eval != "unk":
                if f1_eval > best_f1:
                    best_f1 = f1_eval
                    torch.save(model.state_dict(), os.path.join(
                        ConfigClass.trained_dir, 'model_{}.pth'.format(epoch)))


def eval(model, dataloader):
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
        g, lb = dataloader[i]
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
    with open('eval_dict.json', 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(out_dict)
    return mean_loss.avg, mean_acc.avg, f1_meter.get()['aux_f1']


if __name__ == '__main__':
    # loaddataset
    dataloader = default_dataset
    # model
    meta_graph = [('cfg', 'cfglink_for', 'cfg'),
            ('cfg', 'cfglink_back', 'cfg'),
            ('cfg', 'cfg_passT_link', 'passing_test'),
            ('passing_test', 'passT_cfg_link', 'cfg'),
            ('cfg', 'cfg_failT_link', 'failing_test'),
            ('failing_test', 'failT_cfg_link', 'cfg')]

    model = HeteroMPNNPredictor(default_dataset.cfg_label_feats,
                                default_dataset.cfg_content_feats,
                                256, 32, meta_graph, 2, device)
    train(model, dataloader, ConfigClass.n_epochs)
