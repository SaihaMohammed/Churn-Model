import os
#Directory paths
SRC_DIRECTORY=os.path.split(os.path.realpath(__file__))[0]
ROOT_DIRECTORY=os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY=os.path.join(ROOT_DIRECTORY,'data')
INTERIM_DATA_DIRECTORY=os.path.join(DATA_DIRECTORY,'interim')
RAW_DATA_DIRECTORY=os.path.join(DATA_DIRECTORY,'raw')
PROCESSED_DATA_DIRECTORY=os.path.join(DATA_DIRECTORY,'data')

#File paths - Raw data
RAW_DATA_PATH= os.path.join(RAW_DATA_DIRECTORY,"WA_Fn-UseC_-Telco-Customer-Churn.csv")


#FILE PATHS
X_TRAIN_RAW_PATH=os.path.join(RAW_DATA_DIRECTORY,"X_train.csv")
X_TEST_RAW_PATH=os.path.join(RAW_DATA_DIRECTORY,"X_test.csv")
Y_TRAIN_RAW_PATH=os.path.join(RAW_DATA_DIRECTORY,"y_train.csv")
Y_TEST_RAW_PATH=os.path.join(RAW_DATA_DIRECTORY,"y_test.csv")

#File paths cleaned train data
X_TRAIN_CLEAN_PATH =os.path.join(INTERIM_DATA_DIRECTORY,'X_train.csv')
Y_TRAIN_CLEAN_PATH = os.path.join(INTERIM_DATA_DIRECTORY,'y_train.csv')
