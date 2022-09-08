import json
import os
from abc import ABC, abstractstaticmethod

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams
from torch_geometric.utils import remove_isolated_nodes, to_networkx

from data import ASOSData_pyg

# rcParams['font.family'] = "serif"
rcParams["mathtext.fontset"] = "cm"

matplotlib.rc("xtick", labelsize=12)
matplotlib.rc("ytick", labelsize=12)


class ReportSection(ABC):
    @property
    def header(self):
        return "## " + self._header

    @property
    @abstractstaticmethod
    def body(self):
        pass

    def __str__(self):
        return self.header + "\n" + self.body + "\n"


class ReportConfig(ReportSection):
    def __init__(self, path, config_file):
        self._header = "Config"
        self._config = config_file
        self._path = path

    @property
    def body(self):
        string = json.dumps(self._config, indent=True)
        return "\n" + string + "\n"


class ReportModel(ReportSection):
    def __init__(self, path, model):
        self._header = "Model"
        self._model = model

    @property
    def body(self):
        self._body = "#### " + self._model.name + "\n"
        self._body += str(self._model.describe())
        return self._body + "\n"


class ReportData(ReportSection):
    def __init__(self, path, data):
        self._header = "Data"
        self._dataset_name = data.name
        self._data_desc = data.describe()

        # if isinstance(data.data, ASOSData_pyg):
        #     self.analyse_subgraphs(data.data[0])

    def analyse_subgraphs(self, dataset):
        hg = dataset.to_homogeneous()

        g = to_networkx(hg, to_undirected=True)
        sg = g.subgraph(
            list(range(200)) + list(map(lambda x: x + 150000, list(range(200))))
        )

        components = list(
            nx.connected_components(g)
        )  # list because it returns a generator

        components.sort(key=len, reverse=True)

        # for i in range(len(components)):
        #     if len(components[i]) == 1:
        #         print(i)
        #         one_sig = i
        #         break

        print("no.nodes = " + str(g.order()))
        print("no. subgraphs = " + str(len(components)))
        # iso = len(components)-one_sig
        # print("no. isolated nodes = "+str(iso))

        # S = [g.subgraph(c).copy() for c in nx.connected_components(g)]

        # for i in range(10):
        #     print(S[i])
        #     print('degree of each node in graph = '+str(nx.average_degree_connectivity(S[i])))

    @property
    def body(self):
        self._body = "### " + self._dataset_name + "\n\n"
        self._body += str(self._data_desc)
        return self._body


class ReportTraining(ReportSection):
    def __init__(self, path, scores, save=False):
        self._path = path
        self._plot_path = os.path.join(path, "plots")

        if not os.path.exists(self._plot_path):
            os.mkdir(self._plot_path, mode=0o777)

        self._header = "Results"
        self._body = ""
        self._train_scores = scores["train-scores"]
        self._val_scores = scores["val-scores"]

        # try:
        self.write_training_scores()

        self.plot_losses()
        self.plot_prec_rec()
        self.plot_f1()
        # except:
        #     pass

    def write_training_scores(self):
        for epoch, (
            loss,
            val_loss,
            prec,
            val_prec,
            recall,
            val_recall,
            f1,
            val_f1,
        ) in enumerate(
            zip(
                self._train_scores["total_losses"],
                self._val_scores["total_losses"],
                self._train_scores["precision"],
                self._val_scores["precision"],
                self._train_scores["recall"],
                self._val_scores["recall"],
                self._train_scores["f1-score"],
                self._val_scores["f1-score"],
            )
        ):
            self._body += f"\nEpoch {epoch + 1}: \n"
            self._body += f"Training   - Loss: {loss:.3f}, Precision: {prec:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}\n"
            self._body += f"Validation - Loss: {val_loss:.3f}, Precision: {val_prec:.3f}, Recall: {val_recall:.3f}, F1-score: {val_f1:.3f}\n"

    def plot_losses(self):
        fig, axes = plt.subplots(1, 3, figsize=(25, 8))

        axes[0].plot(
            self._train_scores["total_losses"],
            label="Training",
            color="firebrick",
            lw=2,
        )
        axes[0].plot(
            self._val_scores["total_losses"],
            label="Validation",
            color="blue",
            ls="--",
            lw=2,
        )
        axes[1].plot(
            self._train_scores["clf_losses"], label="Training", color="firebrick", lw=2
        )
        axes[1].plot(
            self._val_scores["clf_losses"],
            label="Validation",
            color="blue",
            ls="--",
            lw=2,
        )
        axes[2].plot(
            self._train_scores["reg_losses"], label="Training", color="firebrick", lw=2
        )
        axes[2].plot(
            self._val_scores["reg_losses"],
            label="Validation",
            color="blue",
            ls="--",
            lw=2,
        )

        plt.legend(fontsize=14)
        # plt.title("Training and Validation CE Losses over 100 epochs.")
        axes[0].set_xlabel("Epoch", fontsize=18)
        axes[0].set_ylabel("Total Loss", fontsize=18)

        axes[1].set_xlabel("Epoch", fontsize=18)
        axes[1].set_ylabel("Classification Loss", fontsize=18)

        axes[2].set_xlabel("Epoch", fontsize=18)
        axes[2].set_ylabel("Regularisation Loss", fontsize=18)

        axes[0].grid(True)
        axes[1].grid(True)
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self._plot_path, "Loss-full.png"))

    def plot_prec_rec(self):
        fig = plt.figure(figsize=(10, 8))

        plt.plot(
            self._train_scores["precision"],
            label="Training Precision",
            color="firebrick",
            lw=2,
        )
        plt.plot(
            self._val_scores["precision"],
            label="Validation Precision",
            color="blue",
            lw=2,
        )
        plt.plot(
            self._train_scores["recall"],
            label="Training Recall",
            color="firebrick",
            ls="--",
            lw=2,
            alpha=0.8,
        )
        plt.plot(
            self._val_scores["recall"],
            label="Validation Recall",
            color="blue",
            ls="--",
            lw=2,
            alpha=0.8,
        )

        plt.legend(fontsize=14)
        # plt.title("Training and Validation Precision & Recall over 100 epochs.")
        plt.xlabel("Epoch", fontsize=18)
        plt.ylabel("Precision & Recall", fontsize=18)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self._plot_path, "PR-full.png"))

    def plot_f1(self):
        fig = plt.figure(figsize=(10, 8))

        plt.plot(
            self._train_scores["f1-score"], label="Training", color="firebrick", lw=2
        )
        plt.plot(
            self._val_scores["f1-score"],
            label="Validation",
            color="blue",
            ls="--",
            lw=2,
        )

        plt.legend(fontsize=14)
        # plt.title("Training and Validation F1-scores over 100 epochs.")
        plt.xlabel("Epoch", fontsize=18)
        plt.ylabel("F1-score", fontsize=18)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self._plot_path, "F1-full.png"))

    @property
    def body(self):
        if isinstance(self._train_scores["accuracy"], list):
            self._body += "#### Final Train Scores\n"
            self._body += f"Accuracy: {100*self._train_scores['accuracy'][-1]:.2f}%\n"
            self._body += f"F1-score: {self._train_scores['f1-score'][-1]:.3f}\n"
            self._body += f"Coverage: {100*self._train_scores['coverage'][-1]:.2f}%\n"
            self._body += "#### Final Validation Scores\n"
            self._body += f"Accuracy: {100*self._val_scores['accuracy'][-1]:.2f}%\n"
            self._body += f"F1-score: {self._val_scores['f1-score'][-1]:.3f}\n"
            self._body += f"Coverage: {100*self._val_scores['coverage'][-1]:.2f}%\n"
        else:
            self._body += "#### Final Train Scores\n"
            self._body += f"Accuracy: {100*self._train_scores['accuracy']:.2f}%\n"
            self._body += f"F1-score: {self._train_scores['f1-score']:.3f}\n"
            self._body += f"Coverage: {100*self._train_scores['coverage']:.2f}%\n"
            self._body += "#### Final Validation Scores\n"
            self._body += f"Accuracy: {100*self._val_scores['accuracy']:.2f}%\n"
            self._body += f"F1-score: {self._val_scores['f1-score']:.3f}\n"
            self._body += f"Coverage: {100*self._val_scores['coverage']:.2f}%\n"

        return self._body


class ReportResults(ReportSection):
    def __init__(self, path, scores, save=False):
        self._header = "Results"
        self._test_scores = scores["test-scores"]
        self._plot_path = os.path.join(path, "plots")

        self.plot_roc()

    def plot_roc(self):
        fpr = self._test_scores["roc"]["fpr"]
        tpr = self._test_scores["roc"]["tpr"]
        roc_auc = self._test_scores["roc"]["auc"]

        fig = plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color="red", label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=1,
            linestyle="--",
            label="Random Classifier",
        )

        plt.legend(fontsize=15)
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.title("Receiver Operating Characteristic", fontsize=15)
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self._plot_path, "roc_curve.png"))

    @property
    def body(self):
        self._body = "#### Final Test Scores\n"
        self._body += f"Accuracy: {100*self._test_scores['accuracy']:.2f}%\n"
        self._body += f"F1-score: {self._test_scores['f1-score']:.3f}\n"
        self._body += f"Coverage: {100*self._test_scores['coverage']:.2f}%\n"
        return self._body


class ReportFooter(ReportSection):
    def __init__(self, path):
        self._header = "End of Report"

    @property
    def body(self):
        self._body = "--END--"
        return self._body
