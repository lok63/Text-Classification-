# Text-Classification-

The project contains 2 parts:
1. Notebooks
Here you can find the pre-processing steps and how i implemented the model. At the end of the notebook you can find a NN implementation as well

2. Packages
Here you can find a package that implements a classification model. You can train, test and export the model in this repository. In addition the packaged has been packed and can be imported locally (i.e import classification_model)
The ml_api implements a flask application which exposes the predict endpoint and uses the pipeline from classification_model to make predictions.


## How to run locally

Please clone the repository and navigate to the root directory.
Create and activate a virtual environment. You can use conda or virtualenv. Make sure you are using python3.

##### Install dependencies

    pip install -r requirements.txt
    

 <b> ** NOTET ** </b>
 There is a chance that when you run the following commands, you will get a NoModuleFound "module_name". If this happens please navigate to that specific directory and paste the following commands on your terminal

##### Unix based system
    export PYTHONPATH='pwd'

##### Windows
    set PYTHONPATH='pwd'   

## 1. Classification model with pipelines

##### Train a new model:
  
      python packages/classification_model/classification_model/train_pipeline.py

#### Test the classification_model package
  
      pytest packages/classification_model/testD


## 2. API - Flask


##### Run the flask API:

    cd packages/ml_api
    export FLASK_APP=run.py
    python run.py


Now you can make post call to the api on the following endpoint:

    127.0.0.1:5000/v1/predict/classification

You can use postman or run this command on the terminal


    curl --header "Content-Type: application/json" --request POST --data '{"message": "Where is my product? It's been a week"}' 127.0.0.1:5000/v1/predict/classification


#### Test the api package

    pytest packages/ml_api/tests




## 3. Deploynment
I would have dockenarise the solution and deploy this on Heroku or AWS EC2 instance. However since this is a small api i could use AWS lambda functions that reduce cost since its an event based system. If the final solution doesnt have a lot of traffic and we want to make on prediction on time then a serverless application is the best way to go. In addition there is a tool called zappa which makes it really easy to deploy the flask API on AWS lambda and add stages such as dev or production.
Furthermore, since this application has been packaged it can be integrared with CircleCI for CI/CD with github.








