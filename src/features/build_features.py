
import os
import pickle
import sys
sys.path.append(".")

import click
import pandas as pd


from src.localpath import *
from src.data.make_dataset import *
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor



@click.group()
def cli():
    pass

def featurize_X(X,predict=False):
    '''Applies featurization to X_train
    '''
    X= drop_customer_id(X)
    X= transform_binary_categorical(X)
    X=one_hot_encode_categorical_features(X,predict=predict)
    X=drop_high_vif_features(X)

    return X


@cli.command()
def create_featurized_data():
    ''' Creates X and y ready for data modelloing and saves
    it to processd data'''
    print("loading data")
    X_train,y_train = load_training_data(clean=True)

    print('featurizing data')
    X_train=featurize_X(X_train)
    y_train= transform_target(y_train)

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

def one_hot_encode_categorical_features(X_train, save_encoder=True,predict=False):
    '''  One hot encodes the categorical features, adds these to the training data, then drops the original columns.
    returns the transformed X_train Data as a pandas dataframes'''
    ohe_filepath=os.path.join(SRC_FEATURES_DIRECTORY,"one_hot_encoder.pkl")
    cols_to_one_hot_encode=X_train.dtypes[X_train.dtypes=='object'].index
    if predict:
        with open(ohe_filepath,"rb") as f :
            ohe=pickle.load(f)
    else:
        ohe=OneHotEncoder(drop="first",sparse=False)
        ohe.fit(X_train[cols_to_one_hot_encode])

    ohe_features=ohe.transform(X_train[cols_to_one_hot_encode])
    ohe_feature_names=ohe.get_feature_names(cols_to_one_hot_encode)
    ohe_df=pd.DataFrame(
        ohe_features,
        columns=ohe_feature_names
    )
    X_train=X_train.assign(**ohe_df)
    X_train=X_train.drop(columns=cols_to_one_hot_encode)

    if save_encoder and not predict:
        ohe_filepath=os.path.join(SRC_FEATURES_DIRECTORY,"one_hot_encoder.pkl")
        print("pickling one_hot encoder")
        with open (ohe_filepath,'wb')as f:
            pickle.dump(ohe,f)

    return X_train
def drop_high_vif_features(X_train):
    '''drops columns with vif greater than 10
    '''
    finished=False
    while not finished:
        vifs=[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
        high_vifs=sorted(zip (X_train.columns,vifs),key=lambda x: x[1], reverse=True)
        high_vif_col,high_vif_value=high_vifs[0]
        if high_vif_value>=10:
            print(f"dropping column {high_vif_col} with vif value of {high_vif_value:.1f}")
            
            X_train=X_train.drop(columns=[high_vif_col])
        else:
            print("Finished dropping columns")
            finished=True


    return X_train
    
def transform_target(y_train):
    y_train['Churn']=y_train['Churn'].map({'Yes':1,'No':0})
    return y_train




if __name__ == "__main__":
    cli()
    

