import numpy as np
import pandas as pd
import statsmodels.api as sm
#import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
#import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
import pickle
import logger

#log_writer = logger.App_Logger()
#file_object = open("logs/ModelTrainingLog.txt", 'a+')

def get_data(log_writer, file_object):
    log_writer.log(file_object, 'Getting the data')
    dta = sm.datasets.fair.load_pandas().data
    return dta

def check_data(data, log_writer, file_object):
    log_writer.log(file_object, 'Checking data parameters')
    print(data.head())
    print(type(data))
    print(data.shape)
    print(data.describe())
    print(data.info())
    print(data.columns)
    print(data.isnull().sum())
    print()

def drop_column(data,col, log_writer, file_object):

    log_writer.log(file_object, 'Deleting the columns {} from dataframe'.format(col))
    data.drop(columns = col, axis = 1, inplace = True)
    return data

def transform_data(data, log_writer, file_object):
    log_writer.log(file_object, 'Appying standard scaling to data')
    scalar = StandardScaler()
    data_scaled = scalar.fit_transform(data)
    return data_scaled, scalar

def pca_data(data, log_writer, file_object):
    log_writer.log(file_object, 'Applying PCA to data')
    pca = PCA(n_components=10)
    new_data = pca.fit_transform(data)

    principal_x = pd.DataFrame(new_data,
                                columns=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9',
                                         'PC-10'])
    return principal_x, pca

def preprocess_data(data, log_writer, file_object):
    log_writer.log(file_object, 'Preprocessing of the data started')
    data['affair'] = (data.affairs > 0).astype(int)
    data = drop_column(data, ['affairs','age'],log_writer, file_object)
    check_data(data, log_writer, file_object)
    y, X = dmatrices(
        'affair ~ rate_marriage + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)', data,
        return_type="dataframe")
    X = X.rename(columns=
                 {'C(occupation)[T.2.0]': 'occ_2',
                  'C(occupation)[T.3.0]': 'occ_3',
                  'C(occupation)[T.4.0]': 'occ_4',
                  'C(occupation)[T.5.0]': 'occ_5',
                  'C(occupation)[T.6.0]': 'occ_6',
                  'C(occupation_husb)[T.2.0]': 'occ_husb_2',
                  'C(occupation_husb)[T.3.0]': 'occ_husb_3',
                  'C(occupation_husb)[T.4.0]': 'occ_husb_4',
                  'C(occupation_husb)[T.5.0]': 'occ_husb_5',
                  'C(occupation_husb)[T.6.0]': 'occ_husb_6'})
    Y = np.ravel(y)
    X = drop_column(X, ['Intercept'], log_writer, file_object)
    X_scaled, scalar = transform_data(X, log_writer, file_object)
    log_writer.log(file_object, 'Scaling of data completed')
    principal_x, pca = pca_data(X_scaled, log_writer, file_object)
    log_writer.log(file_object, 'PCA transformation of data completed')
    return principal_x, Y, scalar, pca

def metrics_data(Y_test,Y_pred, log_writer, file_object):

    log_writer.log(file_object, 'Metrics Calculation Started')
    accuracy1 = accuracy_score(Y_test, Y_pred)
    print("Accuracy is  ",accuracy1)
    log_writer.log(file_object, 'Accuracy is {}'.format(accuracy1))
    conf_mat1 = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix ")
    print(conf_mat1)
    TP = conf_mat1[0][0]
    FP = conf_mat1[0][1]
    FN = conf_mat1[1][0]
    TN = conf_mat1[1][1]
    precision1 = TP / (TP + FP)
    print("Precision is ",precision1)
    log_writer.log(file_object, 'Precision is {}'.format(precision1))
    recall1 = TP / (TP + FN)
    print("Recall is    ",recall1)
    log_writer.log(file_object, 'Recall is {}'.format(recall1))
    f1_score1 = 2 * precision1 * recall1 / (precision1 + recall1)
    print("F1 Score is  " , f1_score1)
    log_writer.log(file_object, 'F1 Score is {}'.format(f1_score1))
    auc1 = roc_auc_score(Y_test, Y_pred)
    print("AUC is   ",auc1)
    log_writer.log(file_object, 'AUC is {}'.format(auc1))
    print()

def save_model(lg, scalar, pca, log_writer, file_object):

    # Writing different model files to file
    log_writer.log(file_object, 'Saving the models at location')
    with open('models/modelForPrediction.sav', 'wb') as f:
        pickle.dump(lg, f)

    with open('models/standardScalar.sav', 'wb') as f:
        pickle.dump(scalar, f)

    with open('models/modelpca.sav', 'wb') as f:
        pickle.dump(pca, f)


def train_data(log_writer):
    file_object = open("logs/TrainingLogs.txt", 'a+')
    log_writer.log(file_object, 'Start of Training')
    data = get_data(log_writer, file_object)
    log_writer.log(file_object, 'Received the data')
    check_data(data, log_writer, file_object)
    X,Y, scalar, pca = preprocess_data(data, log_writer, file_object)
    log_writer.log(file_object, 'Preprocessing of data completed')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=355)
    log_writer.log(file_object, 'Initializing and Fitting the model')
    lg1 = LogisticRegression()
    lg1.fit(X_train, Y_train)
    Y_pred = lg1.predict(X_test)
    metrics_data(Y_test, Y_pred, log_writer, file_object)
    log_writer.log(file_object, 'metrics calculation completed')
    save_model(lg1, scalar, pca, log_writer, file_object)
    log_writer.log(file_object, 'Saving model completed')
    log_writer.log(file_object, '==========================================================')









