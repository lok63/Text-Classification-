import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from classification_model.config import config


def load_dataset(file_name):
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return _data


def save_pipeline(pipeline_to_persist):
    """Persist the pipeline."""

    save_file_name = 'classification_model.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print('saved pipeline')


def load_pipeline(file_name):
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline
