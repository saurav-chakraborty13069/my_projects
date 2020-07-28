import pickle
import pandas as pd
from logger import App_Logger
from patsy import dmatrices

#log_writer = App_Logger()
#file_object = open("logs/ModelPredictLog.txt", 'a+')

def load_models(log_writer,file_object):
    log_writer.log(file_object, 'Starting to load models')
    with open("models/standardScalar.sav", 'rb') as f:
        scalar = pickle.load(f)

    with open("models/modelForPrediction.sav", 'rb') as f:
        model = pickle.load(f)

    with open("models/modelpca.sav", 'rb') as f:
        pca = pickle.load(f)

    return scalar, model, pca


def validate_data(dict_pred, log_writer,file_object):
    # rate = [1.0, 2.0, 3.0, 4.0, 5.0]
    # rel = [1.0, 2.0, 3.0, 4.0]
    # edu = [9.0, 12.0, 14.0, 6.0, 17.0, 20.0]
    occupation = {'occ_2': 0.0,'occ_3':0.0, 'occ_4':0.0, 'occ_5':0.0, 'occ_6':0.0}
    occupation_husb = {'occ_husb_2': 0.0, 'occ_husb_3': 0.0, 'occ_husb_4': 0.0, 'occ_husb_5': 0.0, 'occ_husb_6': 0.0}
    log_writer.log(file_object, 'Converting data to dataframe')
    data_df = pd.DataFrame(dict_pred, index = [1,])
    final_df = pd.DataFrame(columns=['occ_2',	'occ_3',	'occ_4',	'occ_5',	'occ_6',	'occ_husb_2',
                                     'occ_husb_3',	'occ_husb_4',	'occ_husb_5',	'occ_husb_6',	'rate_marriage',
                                     'yrs_married',	'children',	'religious',	'educ'])
    log_writer.log(file_object, 'Writing to final dataframe')
    final_df['rate_marriage'] = data_df['Rate_marriage']
    final_df['yrs_married'] = data_df['Years_married']
    final_df['children'] = data_df['Children']
    final_df['religious'] = data_df['Religious']
    final_df['educ'] = data_df['Education']

    for key in occupation:
        if int(key[-1]) == int(data_df['Occupation']):
            # print(keys[-1])
            occupation[key] = float(data_df['Occupation'])
        else:
            # print(keys)
            occupation[key] = 0.0

    for key in occupation_husb:
        if int(key[-1]) == int(data_df['Occupation_Husb']):
            # print(keys[-1])
            occupation_husb[key] = float(data_df['Occupation_Husb'])
        else:
            # print(keys)
            occupation_husb[key] = 0.0

    final_df['occ_2'] = occupation['occ_2']
    final_df['occ_3'] = occupation['occ_3']
    final_df['occ_4'] = occupation['occ_4']
    final_df['occ_5'] = occupation['occ_5']
    final_df['occ_6'] = occupation['occ_6']
    final_df['occ_husb_2'] = occupation_husb['occ_husb_2']
    final_df['occ_husb_3'] = occupation_husb['occ_husb_3']
    final_df['occ_husb_4'] = occupation_husb['occ_husb_4']
    final_df['occ_husb_5'] = occupation_husb['occ_husb_5']
    final_df['occ_husb_6'] = occupation_husb['occ_husb_6']

    return final_df


def predict_data(dict_pred, log_writer):

    #validate the data entered
    #preprocess to get X in sme format
    #then apply models to predict
    file_object = open("logs/PredictionLogs.txt", 'a+')
    log_writer.log(file_object, 'Starting the predict data')

    scalar, model, pca = load_models(log_writer,file_object)
    log_writer.log(file_object, 'Loading of models completed')
    final_df = validate_data(dict_pred, log_writer, file_object)
    log_writer.log(file_object, 'Prepared the final dataframe')
    log_writer.log(file_object, 'Preprocessing the final dataframe with scalar and pca transform')
    scaled_data = scalar.transform(final_df)
    pca_data = pca.transform(scaled_data)
    principal_data = pd.DataFrame(pca_data,columns=['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8','PC-9','PC-10'])
    log_writer.log(file_object, 'Predicting the result')
    predict = model.predict(principal_data)
    if predict[0] == 1:
        result = 'Affair'
    else:
        result = 'Non-Affair'
    print(result)
    log_writer.log(file_object, 'Prediction completed')
    log_writer.log(file_object, '=================================================')
    return result


#
# rate_marriage = 3.0
# yrs_married = 9.0
# children = 3.0
# religious = 3.0
# educ = 17.0
# occ = 2.0
# occ_hsb = 5.0
#
# mydict  = {'Rate_marriage': rate_marriage, 'Years_married': yrs_married, 'Children': children,
#                   'Religious': religious, 'Education': educ, 'Occupation': occ, 'Occupation_Husb': occ_hsb}
#
#
# predict_data(mydict)