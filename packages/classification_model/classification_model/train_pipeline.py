import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.externals import joblib

from classification_model import pipeline
from classification_model.processing.data_management import (load_dataset, save_pipeline)
from classification_model.config import config


def run_training():
    """Train the model."""

    # read training data
    data = load_dataset(config.TRAINING_DATA_FILE)
    print("------------")
    # select only the customer conversations
    data = data[data["message_source"] == "customer"][["message","case_type"]]


    # divide train and test
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=config.SEED)

    X = data.drop(config.TARGET, axis=1)
    y = data[config.TARGET].apply(lambda x: 0 if x=="cancel_order" else 1)


    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    X_train = X_train[config.FEATURES]
    X_test = X_test[config.FEATURES]

    print(X_train.iloc[0])


    pipeline.full_pipe.fit(X_train, y_train)

    save_pipeline(pipeline_to_persist=pipeline.full_pipe)


if __name__ == '__main__':
    run_training()
