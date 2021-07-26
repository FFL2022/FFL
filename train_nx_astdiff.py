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
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = {}
        self.tn = {}
        self.fp = {}
        self.fn = {}
        for i in range(self.num_classes):
            self.tn[i] = 0
            self.tp[i] = 0
            self.fp[i] = 0
            self.fn[i] = 0

    def update(self, cal, labels):
        for i in range(self.num_classes):
            self.tp[i] += torch.sum((cal[labels == i] == i)).item()
            self.tn[i] += torch.sum((cal[labels != i] != i)).item()
            self.fp[i] += torch.sum((cal[labels != i] == i)).item()
            self.fn[i] += torch.sum((cal[labels == i] != i)).item()

    def get(self):
        out_dict = {}
        for i in range(self.num_classes):
            tnr = 'unk'
            tpr = 'unk'
            prec = 'unk'
            rec = 'unk'
            aux_f1 = 'unk'
            if self.tn[i] + self.tp[i] > 0:
                tnr = self.tn[i]/(self.tn[i] + self.tp[i])
            if (self.tp[i] + self.tn[i]) > 0:
                tpr = self.tp[i]/(self.tp[i] + self.fn[i])
            if (self.tp[i] + self.fp[i]) > 0:
                prec = self.tp[i]/(self.tp[i] + self.fp[i])
            if (self.tp[i] + self.fn[i]) > 0:
                rec = self.tp[i]/(self.tp[i] + self.fn[i])
            if prec != 'unk' and rec != 'unk':
                aux_f1 = (prec + rec)/2
            out_dict[i] = {'tpr': tpr, 'tnr': tnr, 'prec': prec, 'rec': rec,
                           'aux_f1': aux_f1}
        if all(out_dict[i]['aux_f1'] != 'unk' for i in range(self.num_classes)):
            out_dict['aux_f1'] = sum(
                out_dict[i]['aux_f1']
                for i in range(self.num_classes))/self.num_classes
        else:
            out_dict['aux_f1'] = 'unk'


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

    # mean_loss = AverageMeter()
    # mean_acc = AverageMeter()
    mean_ast_loss = AverageMeter()
    mean_ast_acc = AverageMeter()

    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()

    f1_meter = BinFullMeter(3)
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
            _, ast_cal = torch.max(g.nodes['ast'].data['logits'].detach().cpu(),
                                   dim=1)

            preds = g.nodes['ast'].data['pred'][:, 1].detach().cpu()
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


def eval(model, dataloader, epoch):
    dataloader.val()
    # mean_loss = AverageMeter()
    # mean_acc = AverageMeter()
    mean_ast_loss = AverageMeter()
    mean_ast_acc = AverageMeter()
    f1_meter = BinFullMeter(3)
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
        ast_loss = F.cross_entropy(g.nodes['ast'].data['logits'])
        # cfg_loss = F.cross_entropy(logits, lb)
        loss = ast_loss  # + cfg_loss
        preds = g.nodes['ast'].data['pred'][:, 1]
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
        mean_ast_acc.update(torch.sum(ast_cal == ast_lb).item()/g.number_of_nodes('ast'),
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
