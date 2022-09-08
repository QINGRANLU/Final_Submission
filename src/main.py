import argparse
import json
import random

import numpy as np
import torch
import torch_geometric

from config import Parser
from data import (
    DatasetManager,
    SelectLowAndHighReturningCustomers,
    SelectLowAndHighReturningProducts,
)
from models import (
    GNNClf,
    LogisticRegressionClf,
    MetaPath2VecClf,
    MLPClf,
    RandomForestClf,
    XGBoostClf,
)
from reports import ReportManager

LEARNERS = {
    "logistic-regression": LogisticRegressionClf,
    "random-forest": RandomForestClf,
    "xgboost": XGBoostClf,
    "mlp": MLPClf,
    "metapath2vec": MetaPath2VecClf,
    "gnn": GNNClf,
}

if __name__ == "__main__":
    # Take in bash command.
    cli_parser = argparse.ArgumentParser(description="ASOS Return Prediction.")
    cli_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="File containing configuration for experiment.",
        default="config/template.json",
    )
    args = cli_parser.parse_args()

    # Load in the config file.
    if args.config:
        try:
            with open(args.config) as json_file:
                config = json.load(json_file)
        except IOError as e:
            print(
                "Input/Output error when loading in file:\n"
                + f"{0}: {1}".format(e.errno, e.strerror)
            )
    else:
        raise ValueError("No config file provided.")

    # Parse the arguments.
    parser = Parser(config)

    # Unpack the output arguments of the parser.
    exp_args, data_args, model_args, train_args = parser.parse()

    # Create a report.
    report = ReportManager(exp_args["path"])
    # Add configuration to report.
    report.add_section(section="config", config_file=config)
    report.save()

    # Update the random seeds for reproducible results.
    if exp_args["seed"]:
        torch_geometric.seed.seed_everything(exp_args["seed"])

    # Get the training data.
    data = DatasetManager(data_args["type"], data_args["args"])

    if data_args["args"]["customer_transforms"]["lowAndHighReturns"] == "train":
        data.data.customer_transforms.steps.remove(
            ("lowAndHighReturns", data.data.customer_transforms["lowAndHighReturns"])
        )
    elif data_args["args"]["customer_transforms"]["lowAndHighReturns"] == "test":
        data.data.customer_transforms.steps.insert(
            1, ("lowAndHighReturns", SelectLowAndHighReturningCustomers())
        )
    else:
        pass

    if data_args["args"]["product_transforms"]["lowAndHighReturns"] == "train":
        data.data.product_transforms.steps.remove(
            ("lowAndHighReturns", data.data.product_transforms["lowAndHighReturns"])
        )
    elif data_args["args"]["product_transforms"]["lowAndHighReturns"] == "train":
        data.data.product_transforms.steps.insert(
            1, ("lowAndHighReturns", SelectLowAndHighReturningProducts())
        )
    else:
        pass
    print("end get train data")
    print(data)
    # Get the fitted customer transforms ready to be applied to the test data.
    data_args["args"]["customer_transforms"] = data.data.customer_transforms
    data_args["args"]["product_transforms"] = data.data.product_transforms

    # Get the validation data.
    val_data = DatasetManager(data_args["type"], data_args["args"], validation=True)
    print("end get validation data")
    # Get the test data.
    test_data = DatasetManager(data_args["type"], data_args["args"], test=True)
    print("end get test data")
    # Add data description to report.
    report.add_section(section="data", data=data)
    report.save()

    # Models.
    print("before model")
    model = LEARNERS[model_args["type"]](
        data,
        test_data,
        val_dataset=val_data,
        loss=model_args["loss"],
        model_args=model_args["args"],
        path=report.path,
    )

    # Add model description to report.
    report.add_section(section="model", model=model)
    report.save()

    # Training.
    print("before model.train")
    model.train(**train_args)
    print("end model.train")
    # Results.
    train_scores, val_scores = model.get_train_results()

    training_scores = {"train-scores": train_scores, "val-scores": val_scores}

    # Add training description to report.
    report.add_section(section="training", scores=training_scores)
    report.save()

    test_scores = {"test-scores": model.test()}

    # Add results section to report.
    report.add_section(section="results", scores=test_scores)
    report.save()

    data.data.remove_data()
    test_data.data.remove_data()

    report.add_section(section="footer")
    report.save()
