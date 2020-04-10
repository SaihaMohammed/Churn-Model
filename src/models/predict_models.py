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

@click.group()
def cli():
    pass


@cli.command()
@click.option('--file-name',type=str, required=True)
def predict(file_name):
    ''' Predicts churn or not churn for each row of data in file_name.The file name must be comma delimited.
    Column names dont matter,but the order of columns should be same as the order of original data
    '''

#load data
    X = pd.read_csv(file_name)
# clean and featurize data
    X= clean_X(X)
    X=featurize_X(X,predict=True)
    

# make predictions


# Print those predictions

    print(X.columns)






if __name__ == "__main__":
    cli()
