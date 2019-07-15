import math

from classification_model.predict import make_prediction
from classification_model.processing.data_management import load_dataset
import numpy as np
import json

def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')   # Get a single instance



    # When
    subject = make_prediction(input_data=single_test_json)        #Call the clf to make a prediction


    # Then
    assert subject is not None                                    #assert the prediction is not empty
    assert isinstance(subject.get('predictions')[0], np.int64)    #ensure the preduction returns either 0,1 ->int64
    assert math.ceil(subject.get('predictions')[0]) == 0          # We now that the first row preidction is 0


def test_make_multiple_predictions():
    # Given
    test_data = load_dataset(file_name='test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data.to_json(orient='records')

    true_predictions = [0,1,0,1]

    print("#####################")
    print(multiple_test_json)

    # When
    subject = make_prediction(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 4
    for i, pred in enumerate(subject.get('predictions')):
        print(i)
        assert pred == true_predictions[i]
        
