class Parser():
    """
    Argument Parser for config file.

    This object takes in the config file and separates out the relevant arguments
    into their specific categories.

    Argument Categories:
    - experiment_args
    - data_args
    - model_args
    - training_args

    Methods:
    - parse(): Returns all the arguments in one handy function call.
    """
    def __init__(self, config):
        self._config_args = config

    @property
    def experiment_args(self):
        return self._config_args["Experiment"]

    @property
    def data_args(self):
        return self._config_args["Data"]

    @property
    def model_args(self):
        return self._config_args["Model"]

    @property
    def training_args(self):
        return self._config_args["Training"]
    
    def parse(self):
        return self.experiment_args, self.data_args, self.model_args, self.training_args