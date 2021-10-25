import math
import time
import torch


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


class KFullMeter(object):
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
        if all(out_dict[i]['aux_f1'] != 'unk'
               for i in range(self.num_classes)):
            out_dict['aux_f1'] = sum(
                out_dict[i]['aux_f1']
                for i in range(self.num_classes))/self.num_classes
        else:
            out_dict['aux_f1'] = 'unk'
        return out_dict


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
