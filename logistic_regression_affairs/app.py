from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import os

from logger import App_Logger
from logistic import train_data
from prediction import predict_data



app = Flask(__name__)

log_writer = App_Logger()
#file_object = open("logs/ModelPredictLog.txt", 'a+')

@app.route('/', methods = ['GET'])
@cross_origin()
def home_page():
    return render_template("index.html")


@app.route('/predict', methods = ['GET','POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            file_object = open("logs/GeneralLogs.txt", 'a+')
            log_writer.log(file_object, 'Start getting data from UI')
            rate_marriage = float(request.form['rate_marriage'])
            yrs_married = float(request.form['yrs_married'])
            children = float(request.form['children'])
            religious = float(request.form['religious'])
            educ = float(request.form['educ'])
            occupation = float(request.form['occupation'])
            occupation_husb = float(request.form['occupation_husb'])
            log_writer.log(file_object, 'Complete getting data from UI')
            #age = int(request.form['age'])

            mydict = {'Rate_marriage': rate_marriage, 'Years_married': yrs_married, 'Children': children,
                  'Religious': religious, 'Education': educ, 'Occupation': occupation, 'Occupation_Husb': occupation_husb}
            log_writer.log(file_object, 'Passing mydict to prediction.predict_data')
            prediction = predict_data(mydict,log_writer)
            log_writer.log(file_object, 'Received the prediction')
            return render_template('results.html', prediction=prediction)
            log_writer.log(file_object, '=================================================')
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


@app.route('/train', methods = ['POST','GET'])
@cross_origin()
def train():
    #if request.method == 'POST':
    train_data(log_writer)
    return render_template("index.html")



if __name__ == "__main__":
    # clntApp = ClientApi()
    app.run(host='127.0.0.1', port=8001, debug=True)

