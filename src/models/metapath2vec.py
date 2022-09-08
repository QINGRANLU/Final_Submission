import numpy as np
import torch
from torch_geometric.nn import MetaPath2Vec
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

from .model_template import ModelTemplate

METAPATHS = {
    "customer returns product": ("customer", "returns", "product"),
    "product returned_by customer": ("product", "returned_by", "customer")
}

OPTIMIZERS = {
    "sparse-adam": torch.optim.SparseAdam
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MetaPath2VecClf(ModelTemplate):
    def __init__(self, dataset, loss="mse", model_args=None):
        super().__init__("MetaPath2Vec Classifier")

        self.data = dataset.data
        self.loss = loss

        self.metapaths = [
            METAPATHS[path] for path in model_args.pop("metapaths")
        ]

        self.optimizer_args = model_args.pop("optimizer")

        self.model = MetaPath2Vec(self.data.edge_index_dict, 
                                  metapath=self.metapaths, 
                                  **model_args).to(device)

        self.optimizer = OPTIMIZERS[optimizer_args["name"]](
            list(self.model.parameters()), **optimizer_args["args"])

        self.loader = self.model.loader(batch_size=128, shuffle=True)

    def describe(self):
        return self.name

    def decision_function(self, X):
        pass

    def train(self, epochs):
        self.model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(self.loader):
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            self.optimizer.step()

    def get_training_scores(self):
        pass

    def test(self):
        pass

