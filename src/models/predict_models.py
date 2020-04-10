#Standard imports
import click
import os
import pickle
import sys
sys.path.append('.')

import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from src.features.build_features import featurize_X

from src.data.make_dataset import load_training_data,clean_X
from src.localpath import *
from src.models.train_model import load_pickled_model

PICKLED_MODEL_FILENAME="978465313962498816.pkl"


@click.group()
def cli():
    pass



def predict(file_name,proba=False):
    ''' Predicts churn or not churn for each row of data in file_name.The file name must be comma delimited.
    Column names dont matter,but the order of columns should be same as the order of original data
    '''

    #load data
    X = pd.read_csv(file_name)
    # clean and featurize data
    X = clean_X(X)
    X = featurize_X(X,predict=True)
    #load model
    model=load_pickled_model(PICKLED_MODEL_FILENAME)
    
    # make predictions
    if proba:
        predictions=model.predict_proba(X)[:,1]
    else:
        predictions=model.predict(X)

    # return those predictions
    return predictions


@cli.command()
@click.option('--file-name',type=str, required=True)
def click_predict(file_name):
    ''' Predicts churn or not churn for each row of data in file_name.The file name must be comma delimited.
    Column names dont matter,but the order of columns should be same as the order of original data
    '''
    predictions=predict(file_name)
    print(predictions)



if __name__ == "__main__":
    cli()
