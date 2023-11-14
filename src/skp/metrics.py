import numpy as np
import torch
import pytorch_lightning as pl

from torchmetrics import functional as FM
from torchmetrics import Metric
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def _roc_auc_score(t, p):
    return torch.tensor(roc_auc_score(t, p) if len(np.unique(t)) > 1 else 0.5)


class _BaseMetric(Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('p', default=[], dist_reduce_fx=None)
        self.add_state('t', default=[], dist_reduce_fx=None)

    def update(self, p, t):
        self.p.append(p)
        self.t.append(t)

    def compute(self):
        raise NotImplementedError


class AUROC(_BaseMetric):
    """For simple binary classification
    """
    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy() #(N,4)
        t = torch.cat(self.t, dim=0).cpu().numpy() #(N,4)
        auc_dict = {}
        for c in range(p.shape[1]):
            auc_dict[f'auc{c}'] = _roc_auc_score(t==c, p[:,c])
        auc_dict['auc_girl'] = _roc_auc_score(t[t<2] == 0, p[t<2,0])
        auc_dict['auc_boy']  = _roc_auc_score(t[t<2] == 1, p[t<2,1])
        auc_dict['auc_bg'] = auc_dict['auc_girl'] + auc_dict['auc_boy']
        return auc_dict


class Accuracy(_BaseMetric):

    def compute(self): 
        p = torch.cat(self.p, dim=0)
        t = torch.cat(self.t, dim=0)
        return dict(accuracy=(p.argmax(1) == t).float().mean())
