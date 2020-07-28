import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('Admission_Prediction.csv')
data.head()

data['GRE Score'] = data['GRE Score'].fillna(data['GRE Score'].mean())
data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())
data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])

data = data.drop(columns = 'Serial No.')
data.head()

y = data['Chance of Admit']
X =data.drop(columns = ['Chance of Admit'])

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)

regression = LinearRegression()

regression.fit(x_train,y_train)

import pickle
# saving the model to the local file system
filename = 'linear_regression_admission.pickle'
pickle.dump(regression, open(filename, 'wb'))

with open('sandardScalar.sav', 'wb') as f:
    pickle.dump(scaler,f)