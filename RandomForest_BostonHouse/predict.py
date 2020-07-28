import pandas as pd
import pickle

def load_models(log_writer, file_object):
    log_writer.log(file_object, 'Starting to load models')
    with open("models/modelForPrediction.sav", 'rb') as f:
        model = pickle.load(f)
    with open("models/scaler.sav", 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_data(dict_pred, log_writer):
    file_object = open("logs/PredictionLogs.txt", 'a+')
    log_writer.log(file_object, 'Starting the predict data')

    model, scaler = load_models(log_writer, file_object)
    log_writer.log(file_object, 'Loading of models completed')
    final_df = pd.DataFrame(dict_pred, index=[1, ])
    df = scaler.transform(final_df)
    print(type(final_df))
    print(type(df))
    log_writer.log(file_object, 'Prepared the final dataframe')
    log_writer.log(file_object, 'Predicting the result')
    prediction = model.predict(df)

    print('Prediction is:    ', prediction[0])
    log_writer.log(file_object, 'Prediction completed')
    log_writer.log(file_object, '=================================================')
    return prediction[0]


