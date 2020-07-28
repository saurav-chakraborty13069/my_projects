import pickle
import pandas as pd


def load_models():
    with open("models/standardScalar.sav", 'rb') as f:
        scaler = pickle.load(f)

    with open("models/modelForPrediction.sav", 'rb') as f:
        model = pickle.load(f)
    return scaler, model

def get_dummies_value(data_dict,value):

    for key in data_dict:
        if key.split(' ')[1] == value:
            # print(keys[-1])
            data_dict[key] = 1.0
        else:
            # print(keys)
            data_dict[key] = 0.0

    return data_dict


def validate_data(mydict):
    education = {' Bachelors': 11, ' HS-grad': 9, ' 11th': 7, ' Masters': 12, ' 9th': 5,
                 ' Some-college': 10, ' Assoc-acdm': 15, ' Assoc-voc': 16, ' 7th-8th': 4,
                 ' Doctorate': 14, ' Prof-school': 13, ' 5th-6th': 3, ' 10th': 6, ' 1st-4th': 2,
                 ' Preschool': 1, ' 12th': 8}

    native = {'native_country_ Canada':0.0, 'native_country_ China':0.0,
       'native_country_ Columbia':0.0, 'native_country_ Cuba':0.0,
       'native_country_ Dominican-Republic':0.0, 'native_country_ Ecuador':0.0,
       'native_country_ El-Salvador':0.0, 'native_country_ England':0.0,
       'native_country_ France':0.0, 'native_country_ Germany':0.0,
       'native_country_ Greece':0.0, 'native_country_ Guatemala':0.0,
       'native_country_ Haiti':0.0, 'native_country_ Holand-Netherlands':0.0,
       'native_country_ Honduras':0.0, 'native_country_ Hong':0.0,
       'native_country_ Hungary':0.0, 'native_country_ India':0.0,
       'native_country_ Iran':0.0, 'native_country_ Ireland':0.0,
       'native_country_ Italy':0.0, 'native_country_ Jamaica':0.0,
       'native_country_ Japan':0.0, 'native_country_ Laos':0.0,
       'native_country_ Mexico':0.0, 'native_country_ Nicaragua':0.0,
       'native_country_ Outlying-US(Guam-USVI-etc)':0.0, 'native_country_ Peru':0.0,
       'native_country_ Philippines':0.0, 'native_country_ Poland':0.0,
       'native_country_ Portugal':0.0, 'native_country_ Puerto-Rico':0.0,
       'native_country_ Scotland':0.0, 'native_country_ South':0.0,
       'native_country_ Taiwan':0.0, 'native_country_ Thailand':0.0,
       'native_country_ Trinadad&Tobago':0.0, 'native_country_ United-States':0.0,
       'native_country_ Vietnam':0.0, 'native_country_ Yugoslavia':0.0}

    sex = {'Male':1, 'Female':0}

    race = {'race_ Asian-Pac-Islander':0.0, 'race_ Black':0.0, 'race_ Other':0.0, 'race_ White':0.0}

    relationship = {'relationship_ Not-in-family':0.0, 'relationship_ Other-relative':0.0,
       'relationship_ Own-child':0.0, 'relationship_ Unmarried':0.0,
       'relationship_ Wife':0.0}

    occupation = {'occupation_ Armed-Forces':0.0,
       'occupation_ Craft-repair':0.0, 'occupation_ Exec-managerial':0.0,
       'occupation_ Farming-fishing':0.0, 'occupation_ Handlers-cleaners':0.0,
       'occupation_ Machine-op-inspct':0.0, 'occupation_ Other-service':0.0,
       'occupation_ Priv-house-serv':0.0, 'occupation_ Prof-specialty':0.0,
       'occupation_ Protective-serv':0.0, 'occupation_ Sales':0.0,
       'occupation_ Tech-support':0.0, 'occupation_ Transport-moving':0.0}

    marital = {'marital_status_ Married-AF-spouse':0.0,
       'marital_status_ Married-civ-spouse':0.0,
       'marital_status_ Married-spouse-absent':0.0,
       'marital_status_ Never-married':0.0, 'marital_status_ Separated':0.0,
       'marital_status_ Widowed':0.0
               }

    workclass = {'workclass_ Local-gov':0.0,
       'workclass_ Never-worked':0.0, 'workclass_ Private':0.0,
       'workclass_ Self-emp-inc':0.0, 'workclass_ Self-emp-not-inc':0.0,
       'workclass_ State-gov':0.0, 'workclass_ Without-pay':0.0}
    # log_writer.log(file_object, 'Converting data to dataframe')

    data_df = pd.DataFrame(mydict, index=[1, ])
    print(data_df)
    columns = ['education', 'native_country_ Canada', 'native_country_ China',
       'native_country_ Columbia', 'native_country_ Cuba',
       'native_country_ Dominican-Republic', 'native_country_ Ecuador',
       'native_country_ El-Salvador', 'native_country_ England',
       'native_country_ France', 'native_country_ Germany',
       'native_country_ Greece', 'native_country_ Guatemala',
       'native_country_ Haiti', 'native_country_ Holand-Netherlands',
       'native_country_ Honduras', 'native_country_ Hong',
       'native_country_ Hungary', 'native_country_ India',
       'native_country_ Iran', 'native_country_ Ireland',
       'native_country_ Italy', 'native_country_ Jamaica',
       'native_country_ Japan', 'native_country_ Laos',
       'native_country_ Mexico', 'native_country_ Nicaragua',
       'native_country_ Outlying-US(Guam-USVI-etc)', 'native_country_ Peru',
       'native_country_ Philippines', 'native_country_ Poland',
       'native_country_ Portugal', 'native_country_ Puerto-Rico',
       'native_country_ Scotland', 'native_country_ South',
       'native_country_ Taiwan', 'native_country_ Thailand',
       'native_country_ Trinadad&Tobago', 'native_country_ United-States',
       'native_country_ Vietnam', 'native_country_ Yugoslavia', 'sex_ Male',
       'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White',
       'relationship_ Not-in-family', 'relationship_ Other-relative',
       'relationship_ Own-child', 'relationship_ Unmarried',
       'relationship_ Wife', 'occupation_ Armed-Forces',
       'occupation_ Craft-repair', 'occupation_ Exec-managerial',
       'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners',
       'occupation_ Machine-op-inspct', 'occupation_ Other-service',
       'occupation_ Priv-house-serv', 'occupation_ Prof-specialty',
       'occupation_ Protective-serv', 'occupation_ Sales',
       'occupation_ Tech-support', 'occupation_ Transport-moving',
       'marital_status_ Married-AF-spouse',
       'marital_status_ Married-civ-spouse',
       'marital_status_ Married-spouse-absent',
       'marital_status_ Never-married', 'marital_status_ Separated',
       'marital_status_ Widowed', 'workclass_ Local-gov',
       'workclass_ Never-worked', 'workclass_ Private',
       'workclass_ Self-emp-inc', 'workclass_ Self-emp-not-inc',
       'workclass_ State-gov', 'workclass_ Without-pay', 'age', 'fnlwgt',
       'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']


    final_df = pd.DataFrame(columns=columns)
    # log_writer.log(file_object, 'Writing to final dataframe')
    final_df['age'] = data_df['age']
    final_df['fnlwgt'] = data_df['fnlwgt']
    final_df['education_num'] = data_df['education_num']
    final_df['capital_gain'] = data_df['capital_gain']
    final_df['capital_loss'] = data_df['capital_loss']
    final_df['hours_per_week'] = data_df['hours_per_week']



    native_country = get_dummies_value(native,data_df['native_country'][1])
    race = get_dummies_value(race,data_df['race'][1])
    relationship = get_dummies_value(relationship,data_df['relationship'][1])
    occupation = get_dummies_value(occupation,data_df['occupation'][1])
    marital_status = get_dummies_value(marital,data_df['marital_status'][1])
    workclass = get_dummies_value(workclass,data_df['workclass'][1])


    final_df['native_country_ Canada'] = native_country['native_country_ Canada']
    final_df['native_country_ China'] = native_country['native_country_ China']
    final_df['native_country_ Cuba'] = native_country['native_country_ Cuba']
    final_df['native_country_ Dominican-Republic'] = native_country['native_country_ Dominican-Republic']
    final_df['native_country_ Ecuador'] = native_country['native_country_ Ecuador']
    final_df['native_country_ El-Salvador'] = native_country['native_country_ El-Salvador']
    final_df['native_country_ England'] = native_country['native_country_ England']
    final_df['native_country_ France'] = native_country['native_country_ France']
    final_df['native_country_ Germany'] = native_country['native_country_ Germany']
    final_df['native_country_ Greece'] = native_country['native_country_ Greece']
    final_df['native_country_ Guatemala'] = native_country['native_country_ Guatemala']
    final_df['native_country_ Haiti'] = native_country['native_country_ Haiti']
    final_df['native_country_ Holand-Netherlands'] = native_country['native_country_ Holand-Netherlands']
    final_df['native_country_ Honduras'] = native_country['native_country_ Honduras']
    final_df['native_country_ Hong'] = native_country['native_country_ Hong']
    final_df['native_country_ Hungary'] = native_country['native_country_ Hungary']
    final_df['native_country_ India'] = native_country['native_country_ India']
    final_df['native_country_ Iran'] = native_country['native_country_ Iran']
    final_df['native_country_ Ireland'] = native_country['native_country_ Ireland']
    final_df['native_country_ Italy'] = native_country['native_country_ Italy']
    final_df['native_country_ Jamaica'] = native_country['native_country_ Jamaica']
    final_df['native_country_ Japan'] = native_country['native_country_ Japan']
    final_df['native_country_ Laos'] = native_country['native_country_ Laos']
    final_df['native_country_ Mexico'] = native_country['native_country_ Mexico']
    final_df['native_country_ Nicaragua'] = native_country['native_country_ Nicaragua']
    final_df['native_country_ Outlying-US(Guam-USVI-etc)'] = native_country['native_country_ Outlying-US(Guam-USVI-etc)']
    final_df['native_country_ Peru'] = native_country['native_country_ Peru']
    final_df['native_country_ Philippines'] = native_country['native_country_ Philippines']
    final_df['native_country_ Poland'] = native_country['native_country_ Poland']
    final_df['native_country_ Portugal'] = native_country['native_country_ Portugal']
    final_df['native_country_ Puerto-Rico'] = native_country['native_country_ Puerto-Rico']
    final_df['native_country_ Scotland'] = native_country['native_country_ Scotland']
    final_df['native_country_ South'] = native_country['native_country_ South']
    final_df['native_country_ Taiwan'] = native_country['native_country_ Taiwan']
    final_df['native_country_ Thailand'] = native_country['native_country_ Thailand']
    final_df['native_country_ Trinadad&Tobago'] = native_country['native_country_ Trinadad&Tobago']
    final_df['native_country_ United-States'] = native_country['native_country_ United-States']
    final_df['native_country_ Vietnam'] = native_country['native_country_ Vietnam']
    final_df['native_country_ Yugoslavia'] = native_country['native_country_ Yugoslavia']


    final_df['race_ Asian-Pac-Islander'] = race['race_ Asian-Pac-Islander']
    final_df['race_ Black'] = race['race_ Black']
    final_df['race_ Other'] = race['race_ Other']
    final_df['race_ White'] = race['race_ White']

    final_df['relationship_ Not-in-family'] = relationship['relationship_ Not-in-family']
    final_df['relationship_ Other-relative'] = relationship['relationship_ Other-relative']
    final_df['relationship_ Own-child'] = relationship['relationship_ Own-child']
    final_df['relationship_ Unmarried'] = relationship['relationship_ Unmarried']
    final_df['relationship_ Wife'] = relationship['relationship_ Wife']

    final_df['occupation_ Armed-Forces'] = occupation['occupation_ Armed-Forces']
    final_df['occupation_ Craft-repair'] = occupation['occupation_ Craft-repair']
    final_df['occupation_ Exec-managerial'] = occupation['occupation_ Exec-managerial']
    final_df['occupation_ Farming-fishing'] = occupation['occupation_ Farming-fishing']
    final_df['occupation_ Handlers-cleaners'] = occupation['occupation_ Handlers-cleaners']
    final_df['occupation_ Machine-op-inspct'] = occupation['occupation_ Machine-op-inspct']
    final_df['occupation_ Other-service'] = occupation['occupation_ Other-service']
    final_df['occupation_ Priv-house-serv'] = occupation['occupation_ Priv-house-serv']
    final_df['occupation_ Prof-specialty'] = occupation['occupation_ Prof-specialty']
    final_df['occupation_ Protective-serv'] = occupation['occupation_ Protective-serv']
    final_df['occupation_ Sales'] = occupation['occupation_ Sales']
    final_df['occupation_ Tech-support'] = occupation['occupation_ Tech-support']
    final_df['occupation_ Transport-moving'] = occupation['occupation_ Transport-moving']

    final_df['marital_status_ Married-AF-spouse'] = marital_status['marital_status_ Married-AF-spouse']
    final_df['marital_status_ Married-civ-spouse'] = marital_status['marital_status_ Married-civ-spouse']
    final_df['marital_status_ Married-spouse-absent'] = marital_status['marital_status_ Married-spouse-absent']
    final_df['marital_status_ Never-married'] = marital_status['marital_status_ Never-married']
    final_df['marital_status_ Separated'] = marital_status['marital_status_ Separated']
    final_df['marital_status_ Widowed'] = marital_status['marital_status_ Widowed']

    final_df['workclass_ Local-gov'] = workclass['workclass_ Local-gov']
    final_df['workclass_ Never-worked'] = workclass['workclass_ Never-worked']
    final_df['workclass_ Private'] = workclass['workclass_ Private']
    final_df['workclass_ Self-emp-inc'] = workclass['workclass_ Self-emp-inc']
    final_df['workclass_ Self-emp-not-inc'] = workclass['workclass_ Self-emp-not-inc']
    final_df['workclass_ State-gov'] = workclass['workclass_ State-gov']
    final_df['workclass_ Without-pay'] = workclass['workclass_ Without-pay']

    final_df['education'] = data_df['education'].map(education)
    final_df['sex_ Male'] = data_df['sex'].map(sex)

    return final_df


def predict_data(dict_pred):
    scalar, model = load_models()
    # log_writer.log(file_object, 'Loading of models completed')
    final_df = validate_data(dict_pred)
    print(final_df.columns)
    print(final_df.shape)
    # log_writer.log(file_object, 'Prepared the final dataframe')
    # log_writer.log(file_object, 'Preprocessing the final dataframe with scalar and pca transform')

    scaled_data = scalar.transform(final_df)

    # log_writer.log(file_object, 'Predicting the result')
    predict = model.predict(scaled_data)

    print('Class is:    ', predict[0])
    # log_writer.log(file_object, 'Prediction completed')
    # log_writer.log(file_object, '=================================================')
    return predict[0]


# mydict = {'age':39,	'workclass':'State-gov',	'fnlwgt':77516,	'education':'Bachelors',	'education_num':13,
#           'marital_status':'Never-married',	'occupation':'Adm-clerical',	'relationship':'Not-in-family',	'race':'White',
#           'sex':'Male', 'capital_gain':2174,	'capital_loss':0,	'hours_per_week':40,	'native_country':'United-States'	}
#
# predict_data(mydict)