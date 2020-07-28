import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from logger import App_Logger

def get_data(log_writer,file_object):
    log_writer.log(file_object, 'Getting the data')
    from sklearn import datasets
    boston = datasets.load_boston()
    features = pd.DataFrame(boston.data, columns=boston.feature_names)
    targets = boston.target

    return features, targets


def check_data(data,log_writer,file_object):
    print(data.head())
    print(data.columns)
    print(data.info())
    print(data.describe())
    print(type(data))
    print(data.shape)
    print(data.isnull().sum())

def grid_search_data(estimator, x_train,y_train, log_writer,file_object):
    log_writer.log(file_object, 'Setting up the parameters')
    grid_param = {
        "n_estimators": [90, 100, 115],
        'criterion': ['mse', 'mae'],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'min_samples_split': [4, 5, 6, 7, 8],
        'max_features': ['auto', 'log2']
    }
    log_writer.log(file_object, 'Initializing the grid search object')
    grid_search = GridSearchCV(estimator=estimator, param_grid=grid_param, cv=5, n_jobs=-1, verbose=3)
    log_writer.log(file_object, 'Fitting into grid search')
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(best_score)
    return best_params

def save_model(scaler, rand_reg,log_writer,file_object):
    log_writer.log(file_object, 'Saving the models')
    with open('models/modelForPrediction.sav', 'wb') as f:
        pickle.dump(rand_reg, f)

    with open('models/scaler.sav', 'wb') as f:
        pickle.dump(scaler, f)


def train_data(log_writer):
    file_object = open("logs/TrainingLogs.txt", 'a+')
    log_writer.log(file_object, 'Start of Training')

    features, targets = get_data(log_writer,file_object)
    check_data(features, log_writer,file_object)
    log_writer.log(file_object, 'Applying the scaler transform')
    scaler = StandardScaler()
    features_transform = scaler.fit_transform(features)
    log_writer.log(file_object, 'Splitting into train and test')
    x_train, x_test, y_train, y_test = train_test_split(features_transform, targets, test_size=0.30, random_state=355)
    log_writer.log(file_object, 'Intializing a Random forest regressor object')
    rand_clf = RandomForestRegressor(random_state=6)
    log_writer.log(file_object, 'Fitting into regressor and checking the score')
    rand_clf.fit(x_train, y_train)
    print(rand_clf.score(x_test, y_test))
    best_params = grid_search_data(rand_clf,x_train,y_train, log_writer,file_object)
    log_writer.log(file_object, 'Fitting with best parameters')

    rand_reg = RandomForestRegressor(criterion=best_params['criterion'],
                                      max_features=best_params['max_features'],
                                      min_samples_leaf=best_params['min_samples_leaf'],
                                      min_samples_split=best_params['min_samples_split'],
                                      n_estimators=best_params['n_estimators'],
                                      random_state=6)

    rand_reg.fit(x_train, y_train)
    rand_reg.score(x_test, y_test)
    save_model(scaler, rand_reg,log_writer,file_object )
    log_writer.log(file_object, 'End of Training')



