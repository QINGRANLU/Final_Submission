import copy
import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from torch.nn import Linear
from torch_geometric.loader import NeighborLoader

from .model_template import ModelTemplate
from .utils import binary_ce_loss, ce_loss, mse_loss

# from .random_link_transform import RandomLinkSplit

LOSS = {
    "mse": mse_loss,
    "cross-entropy": ce_loss,
    "binary-cross-entropy": binary_ce_loss,
}

OPTIMIZERS = {"sparse-adam": torch.optim.SparseAdam, "adam": torch.optim.Adam}

device = "cuda" if torch.cuda.is_available() else "cpu"


class MLPModel(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()

        self.lins = torch.nn.ModuleList()

        input_layer = Linear(input_channels, hidden_channels[0])
        self.lins.append(input_layer)

        if len(hidden_channels) > 1:
            for n in range(0, len(hidden_channels) - 1):
                lin = Linear(hidden_channels[n], hidden_channels[n + 1])
                self.lins.append(lin)

        self.output_layer = Linear(hidden_channels[-1], 1)

    def forward(self, x_dict, edge_label_index):
        row, col = edge_label_index
        x = torch.cat([x_dict["customer"][row], x_dict["variant"][col]], dim=-1)

        for layer in self.lins:
            x = layer(x)
            x = torch.tanh(x)

        x = self.output_layer(x).sigmoid()
        return torch.cat([x, torch.ones_like(x) - x], dim=1)


class MLPClf(ModelTemplate):
    def __init__(
        self,
        dataset,
        test_dataset,
        val_dataset,
        loss="mse",
        reg_loss=None,
        reg_loss_args=None,
        model_args=None,
        path=None,
    ):
        super().__init__("MLP Classifier")

        self.save_path = path
        self.data = dataset.data if dataset else None
        self.val_data = val_dataset.data if val_dataset else None
        self.test_data = test_dataset.data if test_dataset else None
        self.loss = LOSS[loss]
        self.batch_size = model_args.pop("batch_size")

        optimizer_args = model_args.pop("optimizer")

        self.model = MLPModel(**model_args).to(device)

        self.optimizer = OPTIMIZERS[optimizer_args["name"]](
            list(self.model.parameters()), **optimizer_args["args"]
        )

        self.losses, self.val_losses = [], []

        self.train_losses, self.reg_losses = [], []
        self.val_train_losses, self.val_reg_losses = [], []

        self.accuracy, self.val_accuracy = [], []
        self.precision, self.val_precision = [], []
        self.recall, self.val_recall = [], []
        self.f1, self.val_f1 = [], []

        self.train_dataloader = NeighborLoader(
            self.data.data,
            directed=False,
            # Sample 10 neighbors for each node for 1 iterations
            num_neighbors={key: [10] * 1 for key in self.data.data.edge_types},
            # Use a batch size of 128 for sampling training nodes
            batch_size=self.batch_size,
            input_nodes=("customer", self.data.data["customer"].node_index),
        )

        if val_dataset:
            self.val_dataloader = NeighborLoader(
                self.val_data.data,
                directed=False,
                # Sample 10 neighbors for each node for 1 iterations
                num_neighbors={key: [10] * 1 for key in self.val_data.data.edge_types},
                # Use a batch size of 128 for sampling training nodes
                batch_size=self.batch_size,
                input_nodes=("customer", self.val_data.data["customer"].node_index),
            )

        self.test_dataloader = NeighborLoader(
            self.test_data.data,
            directed=False,
            # Sample 10 neighbors for each node for 1 iterations
            num_neighbors={key: [10] * 1 for key in self.test_data.data.edge_types},
            # Use a batch size of 128 for sampling training nodes
            batch_size=self.batch_size,
            input_nodes=("customer", self.test_data.data["customer"].node_index),
        )

    def describe(self):
        return self.model

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.pt"))

    def load(self, path):
        self.model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"), map_location=torch.device("cpu"))
        )

    def get_train_results(self):
        scores = {
            "total_losses": self.losses,
            "clf_losses": self.train_losses,
            "reg_losses": self.reg_losses,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1-score": self.f1,
        }

        val_scores = {
            "total_losses": self.val_losses,
            "clf_losses": self.val_train_losses,
            "reg_losses": self.val_reg_losses,
            "accuracy": self.val_accuracy,
            "precision": self.val_precision,
            "recall": self.val_recall,
            "f1-score": self.val_f1,
        }

        return scores, val_scores

    def get_data(self):
        return self.train_data, self.val_data

    def decision_function(self, X):
        return self.model.decision_function(X)

    def train(self, epochs, save_epochs=20, patience=None):
        temp_patience = copy.copy(patience)
        for epoch in range(1, epochs + 1):
            loss, train_acc, train_prec, train_rec, train_f1 = 0, 0, 0, 0, 0
            for n, train_data in enumerate(self.train_dataloader, start=1):
                self.model.train()
                self.optimizer.zero_grad()
                train_data.to(device)
                pred = self.model.forward(
                    train_data.x_dict,
                    train_data["customer", "purchases", "variant"].edge_index,
                )

                target = train_data["customer", "purchases", "variant"].edge_label

                train_loss = self.loss(pred, target)
                loss += train_loss

                if epoch % save_epochs == 0:

                    train_results = self.get_scores(pred, target, loss=self.loss)

                    train_acc += train_results["accuracy"]
                    train_prec += train_results["precision"]
                    train_rec += train_results["recall"]
                    train_f1 += train_results["f1-score"]

            loss /= n
            loss.backward()
            self.optimizer.step()

            if epoch % save_epochs == 0:
                val_results = self.validation()

                loss = float(loss.detach().cpu())

                val_loss = float(val_results["clf_loss"].cpu())
                val_acc = val_results["accuracy"]
                val_prec = val_results["precision"]
                val_rec = val_results["recall"]
                val_f1 = val_results["f1-score"]

                self.losses.append(loss)
                self.val_losses.append(val_loss)

                self.accuracy.append(train_acc / n)
                self.val_accuracy.append(val_acc)

                self.precision.append(train_prec / n)
                self.val_precision.append(val_prec)

                self.recall.append(train_rec / n)
                self.val_recall.append(val_rec)

                self.f1.append(train_f1 / n)
                self.val_f1.append(val_f1)

                print(
                    f"\n==================================== \
                    \nEpoch {epoch} \
                    \n------------------------------------ \
                    \nTraining: \
                    \nTotal loss: {self.losses[-1]:.3f} \
                    \nAccuracy:   {100*self.accuracy[-1]:.2f}% \
                    \n------------------------------------ \
                    \nValidation: \
                    \nTotal loss: {self.val_losses[-1]:.3f} \
                    \nAccuracy:   {100*self.val_accuracy[-1]:.2f}%"
                )

                if len(self.val_losses) > 1 and self.val_losses[-1] < min(
                    self.val_losses[:-1]
                ):
                    self.save()

                if patience:
                    if len(self.val_losses) > 1 and self.val_losses[-1] < min(
                        self.val_losses[:-1]
                    ):
                        temp_patience = copy.copy(patience)
                    else:
                        temp_patience -= 1
                        print(temp_patience)

            if patience and temp_patience < 0:
                break

    def get_training_scores(self):
        self.model.eval()
        train_data, val_data = self.get_data()

        target = train_data["customer", "purchases", "variant"].edge_label
        pred = self.model.forward(
            train_data.x_dict, train_data["customer", "purchases", "variant"].edge_index
        )

        scores = self.get_scores(pred, target, loss=self.loss)

        val_scores = self.validation(val_data)

        return scores, val_scores

    @torch.no_grad()
    def validation(self):
        validation_scores = {
            "loss": 0,
            "clf_loss": 0,
            "reg_loss": 0,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
        }

        self.model.eval()
        for n, val_data in enumerate(self.val_dataloader, start=1):
            val_data.to(device)
            pred = self.model.forward(
                val_data.x_dict, val_data["customer", "purchases", "variant"].edge_index
            )

            target = val_data["customer", "purchases", "variant"].edge_label

            val_loss = self.loss(pred, target)

            scores = self.get_scores(pred, target)

            validation_scores["loss"] += val_loss
            validation_scores["clf_loss"] += val_loss
            validation_scores["accuracy"] += scores["accuracy"]
            validation_scores["precision"] += scores["precision"]
            validation_scores["recall"] += scores["recall"]
            validation_scores["f1-score"] += scores["f1-score"]

            if n == 10:
                break

        validation_scores = {
            key: validation_scores[key] / n for key in validation_scores.keys()
        }

        return validation_scores

    @torch.no_grad()
    def test(self, data=False):
        test_scores = {
            "loss": 0,
            "clf_loss": 0,
            "reg_loss": 0,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
        }

        self.model.eval()
        for n, test_data in enumerate(self.test_dataloader, start=1):
            test_data.to(device)
            pred = self.model.forward(
                test_data.x_dict,
                test_data["customer", "purchases", "variant"].edge_index,
            )

            target = test_data["customer", "purchases", "variant"].edge_label

            clf_loss = self.loss(pred, target)

            scores = self.get_scores(pred, target)

            test_scores["loss"] += clf_loss
            test_scores["clf_loss"] += clf_loss
            test_scores["accuracy"] += scores["accuracy"]
            test_scores["precision"] += scores["precision"]
            test_scores["recall"] += scores["recall"]
            test_scores["f1-score"] += scores["f1-score"]

        test_scores = {key: test_scores[key] / n for key in test_scores.keys()}

        roc_scores = self.get_roc_scores(pred[:, 1], target)
        test_scores["roc"] = roc_scores

        return test_scores
