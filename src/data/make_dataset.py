
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
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

    print("Saving data files")
    X_train.to_csv(X_TRAIN_RAW_PATH,index=False)
    X_test.to_csv(X_TEST_RAW_PATH,index=False)
    y_train.to_csv(Y_TRAIN_RAW_PATH,index=False)
    y_test.to_csv(Y_TEST_RAW_PATH,index=False)

def clean_X(X):
    ''' Apply cleaning steps to X
    '''
    diff_values_indx=X[X['TotalCharges']== ' '].index
    X.loc[diff_values_indx,"TotalCharges"]=20
    X['TotalCharges']=X['TotalCharges'].astype(float)

    return X

@cli.command()
def create_clean_train_data():
    '''read Xand Y data ,cleans it and writes it back to data in interim directory 
    '''
    print('loading data')
    X_train,y_train=load_training_data()

    print('cleaning data')
    X_train =clean_X(X_train)

    print('Writing data to interim directory')
    X_train.to_csv(X_TRAIN_CLEAN_PATH,index=False)
    y_train.to_csv(Y_TRAIN_CLEAN_PATH,index=False)


def load_training_data(clean=False,final=False):
    ''' return X_train and y_train if they exist.
    '''
    if clean:
        X_train=pd.read_csv(X_TRAIN_CLEAN_PATH)
        y_train=pd.read_csv(Y_TRAIN_CLEAN_PATH)
    elif final:
        X_train=pd.read_csv(X_TRAIN_FEATURIZED_PATH)
        y_train=pd.read_csv(Y_TRAIN_FEATURIZED_PATH)
    else:
        X_train=pd.read_csv(X_TRAIN_RAW_PATH)
        y_train=pd.read_csv(Y_TRAIN_RAW_PATH)
    
    return X_train,y_train 

def load_testing_data():
    ''' return X_test and y_test if they exist.
    '''
    X_test=pd.read_csv(X_TEST_RAW_PATH)
    y_test=pd.read_csv(Y_TEST_RAW_PATH)
    
    return X_test,y_test



if __name__=="__main__":
    cli()

