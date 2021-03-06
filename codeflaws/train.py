from __future__ import print_function, unicode_literals
import torch
import os
import torch.nn.functional as F
from dataloader import BugLocalizeGraphDataset
from model import HeteroMPNNPredictor
from utils.utils import ConfigClass
from utils.train_utils import BinFullMeter, AverageMeter
import tqdm
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    # model.load_state_dict(torch.load("trained/model_27.pth"))
    # eval(model, dataloader)
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

            mean_loss.update(loss.item(), g.number_of_nodes('cfg'))
            mean_acc.update(
                torch.sum(cal == lb).item()/g.number_of_nodes('cfg'),
                g.number_of_nodes('cfg'))
            f1_meter.update(cal, lb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % ConfigClass.print_rate == 0:
            out_dict = {}
            out_dict['top_1'] = top_1_meter.avg
            out_dict['top_2'] = top_2_meter.avg
            out_dict['top_5'] = top_5_meter.avg
            out_dict['top_10'] = top_10_meter.avg
            out_dict['mean_acc'] = mean_acc.avg
            out_dict['mean_loss'] = mean_loss.avg
            out_dict['f1'] = f1_meter.get()
            with open(ConfigClass.result_dir +
                      '/training_dict_e{}.json'.format(epoch), 'w') as f:
                json.dump(out_dict, f, indent=2)
            print("loss: {}, acc: {}, top 10 acc: {}, ".format(
                mean_loss.avg, mean_acc.avg, top_10_meter.avg) +
                "top 5 acc: {}, top 2 acc {}".format(
                top_5_meter.avg, top_2_meter.avg))
            print(f1_meter.get())
        if epoch % ConfigClass.save_rate == 0:
            l_eval, acc_eval, f1_eval = eval(model, dataloader, epoch)
            if f1_eval != "unk":
                if f1_eval > best_f1:
                    best_f1 = f1_eval
                    torch.save(model.state_dict(), os.path.join(
                        ConfigClass.trained_dir, 'model_{}.pth'.format(epoch)))


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
    with open(ConfigClass.result_dir +
              '/eval_dict_e{}.json'.format(epoch), 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(out_dict)
    return mean_loss.avg, mean_acc.avg, f1_meter.get()['aux_f1']


if __name__ == '__main__':
    # config
    dataset_opt = 'codeflaws'  # nbl, codeflaws
    graph_opt = 2  # 1, 2
    # loaddataset
    data_loader = BugLocalizeGraphDataset(
        dataset_opt=dataset_opt, graph_opt=graph_opt)
    # model
    if graph_opt == 1:
        meta_graph = [('cfg', 'cfglink_for', 'cfg'),
                      ('cfg', 'cfglink_back', 'cfg'),
                      ('cfg', 'cfg_passT_link', 'passing_test'),
                      ('passing_test', 'passT_cfg_link', 'cfg'),
                      ('cfg', 'cfg_failT_link', 'failing_test'),
                      ('failing_test', 'failT_cfg_link', 'cfg')]
        model = HeteroMPNNPredictor(data_loader.cfg_label_feats,
                                    data_loader.cfg_content_feats,
                                    256, 32, meta_graph, 2, device)
    if graph_opt == 2:
        meta_graph = [('cfg', 'cfglink_for', 'cfg'),
                      ('cfg', 'cfglink_back', 'cfg'),
                      ('cfg', 'cfg_passT_link', 'passing_test'),
                      ('passing_test', 'passT_cfg_link', 'cfg'),
                      ('cfg', 'cfg_failT_link', 'failing_test'),
                      ('failing_test', 'failT_cfg_link', 'cfg'),
                      ('ast', 'astlink_for', 'ast'),
                      ('ast', 'astlink_back', 'ast'),
                      ('ast', 'ast_passT_link', 'passing_test'),
                      ('passing_test', 'passT_ast_link', 'ast'),
                      ('ast', 'ast_failT_link', 'failing_test'),
                      ('failing_test', 'failT_ast_link', 'ast'),
                      ('ast', 'ast_cfg_link', 'cfg'),
                      ('cfg', 'cfg_ast_link', 'ast')]
        model = HeteroMPNNPredictor(data_loader.cfg_label_feats,
                                    data_loader.cfg_content_feats,
                                    256, 32, meta_graph, 2, device,
                                    data_loader.ast_label_feats,
                                    data_loader.ast_content_feats)
    ConfigClass.preprocess_dir = "{}/{}/{}".format(
        ConfigClass.preprocess_dir, dataset_opt, graph_opt)
    train(model, data_loader, ConfigClass.n_epochs)
