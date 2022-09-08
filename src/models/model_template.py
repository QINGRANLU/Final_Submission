from abc import ABC, abstractstaticmethod

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from .utils import mse_loss

LOSS = {"mse": mse_loss}


class ModelTemplate(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractstaticmethod
    def describe(self):
        pass

    @abstractstaticmethod
    def train(self):
        pass

    @abstractstaticmethod
    def decision_function(self):
        pass

    def get_scores(self, y_pred, y_true, loss=False):
        scores = {}

        if loss:
            scores["loss"] = loss(y_pred, y_true)

        coverage = self.get_coverage(y_pred, y_true)

        if isinstance(y_pred, torch.Tensor):
            y_pred = np.round(y_pred.detach().cpu().numpy()[:, 1])
        else:
            y_pred = np.round(y_pred[:, 1])

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        scores["accuracy"] = accuracy_score(y_pred, y_true)
        scores["precision"] = precision_score(y_pred, y_true, zero_division=0)
        scores["recall"] = recall_score(y_pred, y_true, zero_division=0)
        scores["f1-score"] = f1_score(y_pred, y_true)
        scores["coverage"] = coverage

        return scores

    def get_coverage(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        high_prob_mask = y_pred[:, 1] > 0.8

        new_preds = np.round(y_pred[high_prob_mask][:, 1])
        new_true = y_true[high_prob_mask]

        tp = (new_preds == 1) & (new_true == 1)
        fp = (new_preds == 1) & (new_true == 0)
        tn = (new_preds == 0) & (new_true == 0)
        fn = (new_preds == 0) & (new_true == 1)

        total = np.sum(y_true == 1)

        return np.sum(tp) / total

    def get_roc_scores(self, y_scores, y_true):
        scores = {}

        if isinstance(y_scores, torch.Tensor):
            y_scores = y_scores.detach().cpu().numpy()

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        scores["tpr"], scores["fpr"] = tpr, fpr
        scores["auc"] = roc_auc

        return scores
