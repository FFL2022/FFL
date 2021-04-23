from __future__ import print_function, unicode_literals

import math
import time
import torch
import torch.nn.functional as F


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


def train(model, train_dataloader, val_dataloader, n_epochs):
    opt = torch.optim.Adam(model.parameters())

    mean_loss = AverageMeter()
    mean_acc = AverageMeter()
    for epoch in range(n_epochs):
        mean_loss.reset()
        mean_acc.reset()
        tn = 0
        tp = 0
        fp = 0
        fn = 0

        model.train()
        for g, lb in train_dataloader:
            g = g.to(device)
            # LB will be preprocessed to have
            lb = lb.to(device)
            model(g)
            # 2 scenario:
            # not using master node
            logits = g.ndata['cfg']['logits']
            # using master node
            loss = F.cross_entropy(logits, lb)
