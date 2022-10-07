# Disaster Response Pipeline Project

### Introduction
This project is to build a web app which can be used to calssify disaster text messages into several categories. Therefore the messages can be routed to correct emergency management agencies. 

The app has an AI engine embedded to classify disastger text messages utilizing Random Forester Classfication algorithm.

### Files Description
1. data folder. This folder has the sample messages and categories in csv format.
   - process_data.py. This python code merges messages and categoreis data, cleans up the data and creates a SQL database.
   - DisasterResponse.db. The database created by python code.
2. models folder. 
   - train_classifier.py. This python code has Random Forester Classification model.
   - classifier.pkl. This file is exported model output by python code.

### Prerequisite
To run python code, the following pythyon packages are needed: pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys, warnings.

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.
   - To run ETL pipeline that cleans data and stores in database 
       `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipelin that trains classifier and saves 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Reference
