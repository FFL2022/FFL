from __future__ import print_function, unicode_literals
import torch
import torch.multiprocessing as mp
import os
import torch.nn.functional as F
from dataloader_nx import CodeflawsFullDGLDataset
from model import HeteroMPNNPredictor1TestNodeType
from train_nx import AverageMeter, BinFullMeter
from utils.utils import ConfigClass
import tqdm
import json
import glob


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, in_queue, out_queue):
    dataset = CodeflawsFullDGLDataset(
        label_mapping_path=ConfigClass.codeflaws_train_cfgidx_map_pkl,
        graph_opt=graph_opt)

    opt = torch.optim.Adam(model.parameters())

    # model.load_state_dict(torch.load("trained/model_27.pth"))
    # eval(model, dataloader)
    done = False
    dataset.train()
    while not done:
        msg, idx = in_queue.get()
        if msg == 'done':
            done = True
            break

        model.train()
        model.zero_grad()
        g = dataset[idx]
        if g is None:
            continue
        # LB will be preprocessed to have
        lb = g.nodes['cfg'].data['tgt']
        non_zeros_lbs = torch.nonzero(lb)
        if non_zeros_lbs.shape[0] == 0:
            continue
        g = g.to(device)
        lbidxs = torch.flatten(non_zeros_lbs).tolist()
        lb = lb.to(device)
        g = model(g)
        # 2 scenario:
        # not using master node
        logits = g.nodes['cfg'].data['logits']
        # using master node, TODO: To be implemented
        loss = F.cross_entropy(logits, lb)
        loss.backward()
        opt.step()

        _, cal = torch.max(logits, dim=1)

        preds = g.nodes['cfg'].data['pred'][:, 1]
        k = min(g.number_of_nodes('cfg'), 10)
        _, indices = torch.topk(preds, k)
        top_10_val = indices[:k].tolist()
        t10f = int(any([idx in lbidxs for idx in top_10_val]))

        k = min(g.number_of_nodes('cfg'), 5)
        top_5_val = indices[:k].tolist()
        t5f = int(any([idx in lbidxs for idx in top_5_val]))

        k = min(g.number_of_nodes('cfg'), 2)
        top_2_val = indices[:k].tolist()
        t2f = int(any([idx in lbidxs for idx in top_2_val]))

        k = min(g.number_of_nodes('cfg'), 1)
        top_1_val = indices[:k].tolist()
        t1f = int(top_1_val[0] in lbidxs)

        loss_item = loss.item()
        acc_item = torch.sum(cal == lb).item()/g.number_of_nodes('cfg')
        cal = cal.cpu().numpy()
        lb = lb.cpu().numpy()
        n = g.number_of_nodes('cfg')
        out_queue.put(('step', (t10f, t5f, t2f, t1f, loss_item, acc_item,
                                cal, lb, n)))


def train_loop(model, dataset, n_workers, epochs=101):
    in_queue, out_queue = mp.Queue(), mp.Queue()
    model.to(device)
    model.share_memory()
    workers = []
    for i in range(n_workers):
        worker = mp.Process(target=train, args=(model, in_queue, out_queue))
        worker.start()
        workers.append(worker)

    mean_loss = AverageMeter()
    mean_acc = AverageMeter()

    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()

    f1_meter = BinFullMeter()
    best_f1 = 0.0

    for epoch in range(epochs):
        mean_loss.reset()
        mean_acc.reset()
        f1_meter.reset()
        top_10_meter.reset()
        top_5_meter.reset()
        top_2_meter.reset()
        top_1_meter.reset()

        for gidx in range(len(dataset)):
            in_queue.put(("step", gidx))

        bar = tqdm.tqdm(range(len(dataset)))
        bar.set_description(f"Train epoch {epoch}")

        for gidx in bar:
            msg, (t10f, t5f, t2f, t1f, loss_item, acc_item, cal, lb, n) =\
                out_queue.get()

            top_10_meter.update(t10f, 1)
            top_5_meter.update(t5f, 1)
            top_2_meter.update(t2f, 1)
            top_1_meter.update(t1f, 1)
            mean_acc.update(acc_item, n)
            mean_loss.update(loss_item, n)
            f1_meter.update(cal, lb)
            bar.set_postfix(loss=loss_item, acc=mean_acc.avg)

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
            print(f"loss: {mean_loss.avg}, acc: {mean_acc.avg}, " +
                  f"top 10 acc: {top_10_meter.avg}, " +
                  f"top 5 acc: {top_5_meter.avg}, " +
                  f"top 2 acc {top_2_meter.avg}," +
                  f"top 1 acc: {top_1_meter.avg}")
            print(f1_meter.get())
        if epoch % ConfigClass.save_rate == 0:
            l_eval, acc_eval, f1_eval = eval(model, dataset, epoch)
            if f1_eval != "unk":
                if f1_eval > best_f1:
                    best_f1 = f1_eval
                    torch.save(model.state_dict(), os.path.join(
                        ConfigClass.trained_dir, f'model_{epoch}_best.pth'))

    for i in range(n_workers):
        in_queue.put(("done", None))
    for worker in workers:
        worker.join()


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
    mp.set_start_method("spawn", force=True)
    # config
    dataset_opt = 'codeflaws'  # nbl, codeflaws
    graph_opt = 2  # 1, 2
    # loaddataset
    dataset = CodeflawsFullDGLDataset(graph_opt=2)
    meta_graph = dataset.meta_graph
    model = HeteroMPNNPredictor1TestNodeType(
        len(ConfigClass.cfg_label_corpus),
        dataset.cfg_content_dim,
        256, 32, meta_graph, 2, device,
        len(dataset.nx_dataset.ast_types),
        dataset.ast_content_dim)
    train_loop(model, dataset, 2, 101)

    ConfigClass.preprocess_dir = "{}/{}/{}".format(
        ConfigClass.preprocess_dir, dataset_opt, graph_opt)
    list_models_paths = list(
        glob.glob(f"{ConfigClass.trained_dir}/model*best.pth"))
    best_latest = max(int(model_path.split("_")[1])
                      for model_path in list_models_paths)
    model_path = f"{ConfigClass.trained_dir}/model_{best_latest}_best.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Evaluation: {model_path}")
    dataset.test()
    eval(model, dataset, best_latest)
