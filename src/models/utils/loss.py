import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

device = "cuda" if torch.cuda.is_available() else "cpu"


def ce_loss(pred, target):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(np.array(pred), dtype=torch.float32)

    if not isinstance(target, torch.Tensor):
        target = torch.tensor(np.array(target), dtype=torch.int64)

    CEloss = CrossEntropyLoss()
    return CEloss(pred, target)


def mse_loss(pred, target):
    return (pred - target).pow(2).mean()


def binary_ce_loss(pred, target):
    pred_squeeze = pred[:, 1].squeeze()
    return F.binary_cross_entropy(pred_squeeze, target.to(torch.float))


def edge_similarity_loss(embs, edge_label_index, targets, delta=0.0):

    return_mask = targets == 1

    return_head, return_tail = edge_label_index[:, return_mask]
    noreturn_head, noreturn_tail = edge_label_index[:, ~return_mask]

    return_edges = torch.cat(
        [embs["customer"][return_head], embs["variant"][return_tail]], dim=1
    )
    noreturn_edges = torch.cat(
        [embs["customer"][noreturn_head], embs["variant"][noreturn_tail]], dim=1
    )

    mean_return = torch.mean(return_edges, dim=0)
    mean_noreturn = torch.mean(noreturn_edges, dim=0)

    sim = torch.nn.CosineSimilarity(dim=0)(mean_return, mean_noreturn)

    non_lin = torch.nn.Sigmoid()

    return non_lin(sim + torch.ones_like(sim))


def cosine_sim(embs, edge_label_index, targets, delta=0.0):
    row, col = edge_label_index

    return_mask = targets == 1

    scores = torch.nn.CosineSimilarity(dim=1)(
        embs["customer"][row], embs["variant"][col]
    )

    scores = scores.unsqueeze(dim=1)
    scores_full = torch.cat([scores, torch.ones_like(scores) - scores], dim=1)

    return binary_ce_loss(scores_full, targets)


def mad_loss(embs, edge_label_index, targets, delta=0.0):

    return_mask = targets == 1

    return_head, return_tail = edge_label_index[:, return_mask]
    noreturn_head, noreturn_tail = edge_label_index[:, ~return_mask]

    return_edges = torch.cat(
        [embs["customer"][return_head], embs["variant"][return_tail]], dim=1
    )
    noreturn_edges = torch.cat(
        [embs["customer"][noreturn_head], embs["variant"][noreturn_tail]], dim=1
    )

    return_dist = torch.mean(torch.pdist(return_edges))
    noreturn_dist = torch.mean(torch.pdist(noreturn_edges))

    inter_dist = torch.mean(torch.cdist(return_edges, noreturn_edges))

    return (return_dist + noreturn_dist) / (2 * inter_dist)


def max_margin_loss(scores, targets, delta=0.0):
    # row, col = edge_label_index

    return_mask = targets == 1

    # scores = torch.nn.CosineSimilarity(dim=1)(
    #     embs["customer"][row], embs["variant"][col]
    # )

    pos_scores = scores[~return_mask, 0]
    neg_scores = scores[~return_mask, 1]

    loss = -torch.mean(pos_scores) + torch.mean(neg_scores) + delta
    # loss = -torch.sum(torch.topk(pos_scores, 50, largest=False)[0]) + torch.sum(torch.topk(neg_scores, 50)[0]) + delta

    non_lin = torch.nn.ReLU()
    return non_lin(loss)
