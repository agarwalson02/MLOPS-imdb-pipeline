import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from src.logger import logging
import yaml
import sys
from src.exception import MyException


def load_model(file_path:str)->pd.DataFrame:
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        raise MyException(f'Failed to parse the CSV file as {e}',sys) from e
    except Exception as e:
        raise MyException(f'Unexpected error occurred while loading the data: {e}',sys) from e

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """the the logistic regression"""
    try:
        clf=LogisticRegression(C=1,solver='liblinear',penalty='l1')
        clf.fit(X_train,y_train)
        logging.info("Model has been successfully trined")
        return clf
    except Exception as e:
        raise MyException(f'Error during model training {e}',sys) from e

def save_model(model,file_path:str)->None:
    """Save the trianed model as file"""
    try:
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logging.info("Model saved to file path %s",file_path)
    except Exception as e:
        raise MyException(f'Error occured while saving file {e}',sys) from e

def main():
    try:
        train_data=load_model("./data/processed/train_bow.csv")
        X_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values

        clf=train_model(X_train,y_train)

        save_model(clf,'models/model.pkl')
    except Exception as e:
        raise MyException(f'Failed to complete the model building process. {e}',sys) from e

if __name__=="__main__":
    main()
