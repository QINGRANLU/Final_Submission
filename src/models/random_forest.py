from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import os
import pickle

from .utils import mse_loss, ce_loss
from .model_template import ModelTemplate

LOSS = {
    "mse": mse_loss,
    "cross-entropy": ce_loss
}

class RandomForestClf(ModelTemplate):
    def __init__(self, dataset=False, test_dataset=False, val_dataset=False, loss="mse", model_args=None, path=None):
        super().__init__("Random Forest Classifier")

        self.save_path = path
        self.data = dataset.data if dataset else None
        self.val_data = val_dataset.data if val_dataset else None
        self.test_data = test_dataset.data if test_dataset else None
        self.loss = LOSS[loss]
        self.model = RandomForestClassifier(**model_args)

    def describe(self):
        return self.model

    def save(self):
        with open(os.path.join(self.save_path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)

    def get_data(self, validation=False):
        if validation:
            return self.data.X, self.val_data.X, self.data.y, self.val_data.y
        else:
            return self.data.X, self.data.y

    def get_test_data(self):
        return self.test_data.X, self.test_data.y

    def decision_function(self, X):
        return self.model.predict_proba(X)[:,1]

    def train(self, epochs):
        X_train, y_train = self.get_data()
        self.model.fit(X_train, y_train)
        self.save()

    def get_train_results(self):
        X_train, X_val, y_train, y_val = self.get_data(validation=True)
        y_train_pred = self.model.predict_proba(X_train)
        y_val_pred = self.model.predict_proba(X_val)

        scores = self.get_scores(y_train_pred, y_train, loss=self.loss)
        val_scores = self.get_scores(y_val_pred, y_val, loss=self.loss)

        return scores, val_scores

    def test(self, data=False):
        X_test, y_test = data if data else self.get_test_data()
        
        y_test_pred = self.model.predict_proba(X_test)

        scores = self.get_scores(y_test_pred, y_test, loss=self.loss)

        roc_scores = self.get_roc_scores(y_test_pred[:,1], y_test)
        scores["roc"] = roc_scores

        return scores
