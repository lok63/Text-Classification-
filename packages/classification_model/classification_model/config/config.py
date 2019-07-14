import pathlib

import classification_model

SEED = 42

PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TRAINING_DATA_FILE = 'tech_test_data-1.csv'
TARGET = 'case_type'


# variables
FEATURES = "message"
