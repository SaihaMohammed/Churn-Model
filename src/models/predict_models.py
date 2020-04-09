#Standard imports
import click
import os
import pickle
import sys
sys.path.append('.')

import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

from src.data.make_dataset import load_training_data
from src.localpath import *

@click.group()
def cli():
    pass


@cli.command()
@click.option('--file-name',type=str, required=True)
def predict(file_name):
    ''' Saves model evaluation mketrics to /model/mode_results.csv and
    saves the pickled model to models.
    '''
    print(file_name)





if __name__ == "__main__":
    cli()
