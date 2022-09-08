import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

from .utils import mse_loss, ce_loss
from .model_template import ModelTemplate

LOSS = {
    "mse": mse_loss,
    "cross-entropy": ce_loss
}

class XGBoostClf(ModelTemplate):
    def __init__(self, dataset=False, test_dataset=False, val_dataset=False, loss="mse", model_args=None, path=None):
        super().__init__("XGBoost Classifier")

        self.save_path = path
        self.data = dataset.data if dataset else None
        self.val_data = val_dataset.data if val_dataset else None
        self.test_data = test_dataset.data if test_dataset else None
        self.loss = LOSS[loss]
        self.early_stopping = model_args.pop("early_stopping_rounds")
        self.model_args = model_args

    def describe(self):
        return self.name

    def save(self):
        self.model.save_model(os.path.join(self.save_path, "model.json"))

    def load(self, path):
        self.model = xgb.Booster()
        self.model.load_model(os.path.join(path, "model.json"))

    def get_data(self, validation=False):
        if validation:
            return self.data.X, self.val_data.X, self.data.y, self.val_data.y
        else:
            return self.data.X, self.data.y

    def get_test_data(self):
        return self.test_data.X, self.test_data.y

    def decision_function(self, X):
        return self.model.predict(X, output_margin=True)

    def train(self, epochs):
        X_train, X_val, y_train, y_val = self.get_data(validation=True)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        evallist = [(dtrain, 'train'), (dval, 'eval')]

        self.model = xgb.train(self.model_args, dtrain, epochs, evallist, early_stopping_rounds=self.early_stopping)
        self.save()

    def get_train_results(self):
        X_train, X_val, y_train, y_val = self.get_data(validation=True)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        y_train_pred = self.model.predict(dtrain)
        y_val_pred = self.model.predict(dval)

        y_train_pred_full = np.array([1 - y_train_pred, y_train_pred]).T
        y_val_pred_full = np.array([1 - y_val_pred, y_val_pred]).T

        scores = self.get_scores(y_train_pred_full, y_train, loss=self.loss)
        scores_val = self.get_scores(y_val_pred_full, y_val, loss=self.loss)

        return scores, scores_val

    def test(self, data=False):
        X_test, y_test = data if data else self.get_test_data()

        dtest = xgb.DMatrix(X_test, label=y_test)

        y_pred_scores = self.model.predict(dtest)

        y_test_pred_full = np.array([1 - y_pred_scores, y_pred_scores]).T

        scores = self.get_scores(y_test_pred_full, y_test, loss=self.loss)

        roc_scores = self.get_roc_scores(y_test_pred_full[:,1], y_test)
        scores["roc"] = roc_scores

        return scores
