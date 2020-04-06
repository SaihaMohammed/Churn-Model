
import os
import sys
sys.path.append(".")

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from src.localpath import *

@click.group()
def cli():
    pass


@cli.command()
def create_train_test_split():
    '''create X and y training set files and holdout set files in the data/raw directory'''
    print('Loading the data')
    df = pd.read_csv(RAW_DATA_PATH)

    print('creating X and y')
    X= df.drop(columns="Churn")
    y= df[["Churn"]]

    print('Creating train-test split')
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

    print("Saving data files")
    X_train.to_csv(X_TRAIN_RAW_PATH,index=False)
    X_test.to_csv(X_TEST_RAW_PATH,index=False)
    y_train.to_csv(Y_TRAIN_RAW_PATH,index=False)
    y_test.to_csv(Y_TEST_RAW_PATH,index=False)



if __name__=="__main__":
    cli()

