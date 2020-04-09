
import os
import pickle
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score,learning_curve
from src.data.make_dataset import load_training_data







def plot_learning_curve (model,X_train,y_train,zoom_out=True):
    ''' This function plots learning curve for the model
    '''
    train_sizes,train_scores,test_scores=learning_curve(model,X_train,y_train)
    train_scores=np.mean(train_scores,axis=1)
    test_scores=np.mean(test_scores,axis=1)

    plt.plot(train_sizes,train_scores,label='Training Accuracy')
    plt.plot(train_sizes,test_scores,label="Testing Accuracy")
    if zoom_out:
        plt.ylim(0,1.05)

    plt.legend()
    plt.show()

    pass