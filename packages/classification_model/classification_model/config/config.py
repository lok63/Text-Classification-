import pathlib

import classification_model

SEED = 42

PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TRAINING_DATA_FILE = 'tech_test_data-1.csv'
TESTING_DATA_FILE = 'test.csv'
TARGET = 'case_type'


# variables
FEATURES = "message"

PIPELINE_NAME = 'classification'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05