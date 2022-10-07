import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['wordnet', 'punkt', 'stopwords'])

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle
import re


def load_data(database_filepath):
    """
       This function is to load data from database created by process.py
       
       Args:
       database_filepath: string, the path of the database
       
       Return:
       X (DataFrame) : Message features
       Y (DataFrame) : Others
       category_names: category names of the DisasterResponse database
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']  # Message Column
    Y = df.iloc[:, 4:]  # Classification label
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    This function is to split the text into tokens
    
    Args:
      text(str): the messages
    
    Return:
      lemm: a list of lemmatized token
    """

    # Normalize the text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenization
    words = word_tokenize(text)

    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]

    # Lemmatization
    lemmatized = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemmatized


def build_model():
    """
     This function is to build a classification model
     
     Args:
     
     Return:
       cv: classification model
    """
    
    # Create a pipeline using Random Forest Classfier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Create Grid search parameters
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 40]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function is to evaluate the randome forest classification model
    
    Args:
    model: the classification model
    X_test: test set messages
    Y_test: test set target
    category_names: list of category names
    
    Return:
    """
    y_pred = model.predict(X_test)
    
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i += 1
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    """
    This function is to save the model
    
    Args:
        model : selected model
        model_filepath (string): the destination the saved model is in
    
    Returns:
    """
    
    file_name = model_filepath
    with open(file_name, "wb") as f:
        pickle.dump(model, f)

    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()