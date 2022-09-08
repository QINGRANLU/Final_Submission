import pandas as pd
from .asos_pyg import ASOSData_pyg
from .asos_sklearn import ASOSData_sklearn

DATALOADERS = {
    "baseline-sklearn": ASOSData_sklearn,
    "metpath2vec-pyg": ASOSData_pyg,
    "metapath2vec-sg": None,
    "gnn-pyg": ASOSData_pyg
}

class DatasetManager():
    """
    Dataset Holder for the different types of data we have available to us. The 
    ASOS Data can be constructed as a graph or as raw features. In all cases, 
    the same preprocessing steps are applied.

    Dataset options:
    - baseline-sklearn: This constructs the data as raw features in an sklearn 
                        style format for baseline models.

    - gnn-pyg:          This constructs the data as a graph using a PyTorch 
                        Geometric defined dataset.

    Methods:
    - name:     This just acesses the dataset name from the internal data.
    - describe: This is used for the report output to access information
                about the data.
    - load:     Undefined at present.
    """
    def __init__(self, data_type, data_args, validation=False, test=False):
        self.data = DATALOADERS[data_type](validation=validation, test=test, **data_args)
        self.data.process()

    @property
    def name(self):
        return self.data.name

    def describe(self):
        return self.data.describe()

    def load(self):
        pass


if __name__ == "__main__":
    data_args = {
        "load_path": "../../raw",
        "reduced": False
    }

    dataset = DatasetManager("baseline-sklearn", data_args)

    print(dataset.data)