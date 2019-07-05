# This script covers the modelling part of the Disaster Response pipeline
# EXAMPLE: python train_classifier.py -db data/DisasterResponse -dt df_clean -m classifier.pkl

# Import libraries
## Data
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import argparse

## NLP
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

## ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump

# CLI Outputs
import warnings
warnings.filterwarnings("ignore")

def load_data(database_name, table_name):
    """Load  a specific database table and return
    features, labels as well as label names.
    Parameters
    ----------
    database_name : string
        database name with folder location
    Returns
    -------
    X: numpy.ndarray
        Messages to train the model on
    y: numpy.ndarray
        Labels of the messages
    labels: list
     The label names
    """
    #engine = create_engine("sqlite:////Users/gunnar.griese/Desktop/python3_env/disaster-response/data/DisasterResponse.db")#.format(database_name))
    engine = create_engine("sqlite:///../{}.db".format(database_name))
    df = pd.read_sql_table("{}".format(table_name), con=engine)
    df = df[df.related.notnull()]
    X = df.message.values
    y = df.iloc[:, 4:].values
    labels = df.columns[4:].tolist()

    return X, y, labels


def tokenize(text):
    """Tokenize the input text.
    Parameters
    ----------
    text : string
        the text for tokenizing
    Returns
    -------
    clean_tokens : list
        a list of tokens
    """

    # Normalize texts
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize texts
    tokens = word_tokenize(text)

    # Remove stop words
    words = [w for w in tokens if w not in stopwords.words("english")]

    # Instantiate Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize and strip
    clean_tokens = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).strip()
        clean_tokens.append(clean_word)

    return clean_tokens


def build_model():
    """Build and optimize model
    Parameters
    ----------
    None
    Returns
    -------
    model: sklearn.pipeline.Pipeline
    The trained model for prediction.
    """

    # Define the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1)),
    ])

    # Set parameters for grid search; uncommenting increases training time
    parameters = {
    #    'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2']
    }

    # Apply grid search on pipeline; adjusting verbose influences CLI output
    model = GridSearchCV(pipeline, param_grid=parameters,
                         cv=2, verbose=3)
    #model = pipeline
    print(type(model))
    return model

def multioutput_classification_report(y_true, y_pred, labels):
    """ Get information on model performance.
    Parameters
    ----------
    y_true : numpy.ndarray
        True labels
    y_true : numpy.ndarray
        Predicted labels
    labels : label names
        Names of the labels
    Returns
    -------
    model: 
        The trained model for prediction.
    """
    for i in range(0, len(labels)):
        print(labels[i])
        print("\tAccuracy: {:.4f}\t\t% Precision: {:.4f}\t\t% Recall: {:.4f}\t\t% F1_score: {:.4f}".format(
            accuracy_score(y_true[:, i], y_pred[:, i]),
            precision_score(y_true[:, i], y_pred[:, i], average='weighted'),
            recall_score(y_true[:, i], y_pred[:, i], average='weighted'),
            f1_score(y_true[:, i], y_pred[:, i], average='weighted')
        ))

def evaluate_model(model, X_test, y_test, labels):
    """Evaluates and prints model performance
    Parameters
    ----------
    model : text classifier
    X_test: numpy.ndarray
        The test data
    y_test: numpy.ndarray
        The test labels
    labels: list
        The category names
    Returns
    -------
    None
    """
    y_pred = model.predict(X_test)
    multioutput_classification_report(y_test, y_pred, labels)
    
def save_model(model, model_name):
    """Save model as a pickle file
    Parameters
    ----------
    model : trained text classifier
        The optimized classifier
    model_name : string
        location where to store the model
    Returns
    -------
    None
    """
    # Export the model to a file
    dump(model, model_name)

def main():

    """
    Execute the above defined functions.
    For further information on input execute 'python train_classifier.py --help'
    in command line.
    """

    # parse command line inputs
    parser = argparse.ArgumentParser(description='Pipeline for modeling that requires the following inputs:')
    parser.add_argument(
        "-db", "--database", help="Database name to fetch data from.", required=True)
    parser.add_argument(
        "-dt", "--database table", help="Table name to fetch data from.", required=True)
    parser.add_argument(
        "-m", "--model name", help="Model name to store pickle file in the format of: model_name.pkl", required=True)
    args = vars(parser.parse_args())

    DATABASE = args.get("database", "")
    DATABASE_TABLE = args.get("database table", "")
    MODEL_NAME = args.get("model name", "")

    print("sqlite:///{}.db".format(DATABASE))

    print("Loading preprocessed data from... DATABASE: {}".format(DATABASE))
    print("Using table... DATABASE_TABLE: {}".format(DATABASE_TABLE))
    X, Y, labels = load_data(DATABASE, DATABASE_TABLE)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    print("Instantiate model...")
    model = build_model()

    print("Train model...")
    model.fit(X_train, y_train)

    print("Evaluate model...")
    evaluate_model(model, X_test, y_test, labels)

    print('Save model as pickle...   MODEL_NAME: {}'.format(MODEL_NAME))
    save_model(model, MODEL_NAME)

    print("Training job is ... DONE!")


if __name__ == '__main__':
    main()