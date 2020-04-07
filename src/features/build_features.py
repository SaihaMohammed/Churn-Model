
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

    print("Saving data")
    X_train.to_csv(X_TRAIN_FEATURIZED_PATH,index=False)
    y_train.to_csv(Y_TRAIN_FEATURIZED_PATH,index=False)



def drop_customer_id(X_train):
    '''create X a
    '''
    X_train = X_train.drop(columns=['customerID'])
    return X_train




if __name__ == "__main__":
    cli()
    

