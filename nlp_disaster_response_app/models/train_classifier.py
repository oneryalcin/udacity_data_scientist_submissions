import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])

import re
import sys
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


def load_data(database_filepath):
    """ 
    Load data from database and create feature and label vectors
    
    Args:
      database_filepath (str): filepath where Sqlite DB is located
    Returns:
      X (pandas.DataFrame): Disaster Tweets as a dataframe,used as features
      Y (pandas.DataFrame): Disaster Response categories as dataframe. Used as labels
      category_names (list): label names for response categories
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('annotated_tweets', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original','genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """ 
    Tokenize a given text
    
    Args:
      text (str): Text to tokenize
    Returns:
      list: List of text tokens
    """
    
    # remove punctiation
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    
    # tokenize into words
    tokens = word_tokenize(text)
    
    # lemmatize to get the root of the word
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens 
                    if tok not in stopwords.words('english')]
    return clean_tokens

def build_model():
    """
    Build machine learning pipeline for disaster response tweets
    
    Returns:
      sklearn.model_selection.GridSearchCV: ML model (and pipeline) to build NLP model 
                                            using GridSearchCV and DecisionTreeClassifier  
    """
    
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize, min_df=0.0001)),
        ('clf', MultiOutputClassifier(estimator=DecisionTreeClassifier(), n_jobs=-1))
    ])
    
    parameters = {
        'vect__max_df': (0.5, 1.0),
        'clf__estimator__max_features': ['log2', None]
    }

    # Use GridSearchCV to find optimum hyperparameters for ML model
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model 

def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Evaluate model performance
    
    Args:
      model (sklearn.model_selection.GridSearchCV): Trained ML model
      X_test (pandas.DataFrame): Test feature set, here these are tweets.
      Y_test (pandas.DataFrame): Test label set, here these are multiple disaster categories
      category_names: Labels for predictions
    Prints:
      str: classification preport and accuracy scores for each label
    Returns:
        None
    """
    
    # Classify the tweets using test set
    Y_pred = model.predict(X_test)
    
    # Convert to pandas dataframe for easier comparison
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
    
    for col in category_names:
        true, predicted  = Y_test[col], Y_pred[col]
        print("Classification report for column ``{}`` \n".format(col),classification_report(true, predicted))
        print("Accuracy score for column ``{}`` is: {}".format(col, accuracy_score(true, predicted)))
        print("f1 score for column ``{}`` is: {}\n\n".format(col, f1_score(true, predicted, average='macro')))

        
def save_model(model, model_filepath):
    """ 
    Save model as python pickle object to disk 
    
    Args:
      model (sklearn.model_selection.GridSearchCV): Trained ML model
      model_dilepath (str): filepath to save pickled ML model
    Returns:
      None
    """

    pickle.dump(model, open('{}'.format(model_filepath), 'wb'))


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