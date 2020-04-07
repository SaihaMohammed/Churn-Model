
import os
import sys
sys.path.append(".")

import click
import pandas as pd


from src.localpath import *
from src.data.make_dataset import *

@click.group()
def cli():
    pass

@cli.command()
def create_featurized_data():
    ''' Creates X and y ready for data modelloing and saves
    it to processd data'''
    print("loading data")
    X_train,y_train = load_training_data(clean=True)

    print('featurizing data')
    X_train= drop_customer_id(X_train)
    X_train= transform_binary_categorical(X_train)

    print("Saving data")
    X_train.to_csv(X_TRAIN_FEATURIZED_PATH,index=False)
    y_train.to_csv(Y_TRAIN_FEATURIZED_PATH,index=False)



def drop_customer_id(X_train):
    '''create X a
    '''
    X_train = X_train.drop(columns=['customerID'])
    return X_train

def transform_binary_categorical(X_train):
    '''transforms binary categorical features into 
    1 ans 0s'''
    X_train['gender']=X_train['gender'].map({'Female':1,'Male':0})
    X_train['Partner']=X_train['Partner'].map({'Yes':1,'No':0})
    X_train['Dependents']=X_train['Dependents'].map({'Yes':1,'No':0})
    X_train['PhoneService']=X_train['PhoneService'].map({'Yes':1,'No':0})
    X_train['PaperlessBilling']=X_train['PaperlessBilling'].map({'Yes':1,'No':0})
    return X_train
    




if __name__ == "__main__":
    cli()
    

