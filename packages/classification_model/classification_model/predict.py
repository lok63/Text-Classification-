import numpy as np
import pandas as pd

from classification_model.processing.data_management import load_pipeline
from classification_model.config import config


pipeline_file_name = 'classification_model.pkl'
_full_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(input_data):
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)
    prediction = _full_pipe.predict(data[config.FEATURES])
    proba = _full_pipe.predict_proba(data[config.FEATURES])
    response = {'predictions': prediction}

    return response
