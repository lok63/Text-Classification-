import numpy as np
import pandas as pd
import json

from classification_model.processing.data_management import load_pipeline
from classification_model.config import config
from classification_model.processing.validation import validate_inputs
from classification_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_full_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(input_data):
    """Make a prediction using the saved model pipeline."""




    # data = pd.read_json(input_data)
    data = pd.DataFrame([input_data])

    validated_data = validate_inputs(input_data=data)
    prediction = _full_pipe.predict(validated_data[config.FEATURES])
    prediction_proba = _full_pipe.predict_proba(validated_data[config.FEATURES])


    proba = np.round(prediction_proba[0][0],decimals = 5) if prediction == 0 else np.round(prediction_proba[0][1],decimals = 5)
    output = prediction


    results = {'predictions': output, 
               'probability': proba,
               'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results
