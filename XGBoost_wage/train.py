import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle

def get_data():
    url_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    train = pd.read_csv(url_train, header = None)
    url_test= 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    test = pd.read_csv(url_test,skiprows = 1, header = None)
    col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                  'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain',
                  'capital_loss', 'hours_per_week', 'native_country', 'wage_class']

    train.columns = col_labels
    test.columns = col_labels
    return train,test

def check_data(data):
    print(data.describe())
    print(data.head())
    print(data.info())
    print(data.columns)
    print(data.shape)
    for i in data.columns:
        if data[i].dtype != 'int64':
            print(i)
            print(data[i].unique())
            print(data[i].value_counts())
            print()

def merge_data(df1,df2):
    return pd.concat([df1,df2],axis = 1)

def dummies(data, column):
    data_transformed = pd.get_dummies(data,prefix = column, prefix_sep = '_')
    data_transformed = data_transformed.iloc[:,1:]
    return data_transformed

def transform_data(data):
    scaler = StandardScaler()
    X_transform = scaler.fit_transform(data)
    return scaler, X_transform


def grid_search_fit_data(x_train, y_train):
    param_grid = {

        # ' learning_rate':[1,0.5,0.1,0.01,0.001],
        'max_depth': [3, 5, 10, 20],
        'n_estimators': [10, 50, 100, 200]

    }

    grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid, verbose=3)

    grid.fit(x_train, y_train)
    best_params = grid.best_params_
    new_model = XGBClassifier(max_depth=best_params['max_depth'], n_estimators=best_params['n_estimators'])
    new_model.fit(x_train, y_train)
    return new_model


def preprocess_data(train,test):
    data = pd.concat([train,test])
    print(data)
    print(data.shape)
    check_data(data)
    data['workclass'] = data['workclass'].replace(' ?', data['workclass'].value_counts().index[0])
    data['occupation'] = data['occupation'].replace(' ?', data['occupation'].value_counts().index[0])
    data['native_country'] = data['native_country'].replace(' ?', data['native_country'].value_counts().index[0])
    X = data.drop(['wage_class'],axis = 1)
    y = data['wage_class']
    wages = {' <=50K.':0, ' >50K.':1, ' <=50K':0, ' >50K':1}
    y = y.map(wages)
    print(y.shape)

    numerical_var = data[['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']]
    categorical_vars = data[
        ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']]

    workclass = dummies(categorical_vars['workclass'], 'workclass')
    marital = dummies(categorical_vars['marital_status'], 'marital_status')
    occupation = dummies(categorical_vars['occupation'], 'occupation')
    relationship = dummies(categorical_vars['relationship'], 'relationship')
    race = dummies(categorical_vars['race'], 'race')
    sex = dummies(categorical_vars['sex'], 'sex')
    native = dummies(categorical_vars['native_country'], 'native_country')

    df = pd.DataFrame()
    workclass_merged = merge_data(workclass, df)
    marital_merged = merge_data(marital, workclass_merged)
    occ_merged = merge_data(occupation, marital_merged)
    rel_merged = merge_data(relationship, occ_merged)
    race_merged = merge_data(race, rel_merged)
    sex_merged = merge_data(sex, race_merged)
    native_merged = merge_data(native, sex_merged)

    education = {' Bachelors': 11, ' HS-grad': 9, ' 11th': 7, ' Masters': 12, ' 9th': 5,
                 ' Some-college': 10, ' Assoc-acdm': 15, ' Assoc-voc': 16, ' 7th-8th': 4,
                 ' Doctorate': 14, ' Prof-school': 13, ' 5th-6th': 3, ' 10th': 6, ' 1st-4th': 2,
                 ' Preschool': 1, ' 12th': 8}

    categorical_vars['education'] = categorical_vars['education'].map(education)
    categorical_merged = merge_data(categorical_vars['education'], native_merged)
    final_merged = merge_data(categorical_merged, numerical_var)
    print(final_merged.columns)
    return final_merged, y

def save_models(scaler, xgboost):
    with open('models/modelForPrediction.sav', 'wb') as f:
        pickle.dump(xgboost, f)

    with open('models/standardScalar.sav', 'wb') as f:
        pickle.dump(scaler, f)

def train_data():
    train,test = get_data()
    check_data(train)
    check_data(test)
    final_df, y = preprocess_data(test, train)
    scaler, X_transformed = transform_data(final_df)
    x_train, x_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.30, random_state=355)
    model = XGBClassifier(objective='binary:logistic')
    model.fit(x_train, y_train)
    new_model = grid_search_fit_data(x_train, y_train)
    y_pred_new = new_model.predict(x_test)
    predictions_new = [round(value) for value in y_pred_new]
    accuracy_new = accuracy_score(y_test, predictions_new)
    print(accuracy_new)
    save_models(scaler,new_model)
    print("===================================================================")



