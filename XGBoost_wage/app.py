from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
import os
# from logger import App_logger
from predict import predict_data
from train import train_data
import pickle


app = Flask(__name__)

#log_writer = App_logger()

@app.route('/',methods = ['GET'])
@cross_origin()
def home_page():
    return render_template('index.html')


@app.route('/train',methods = ['GET', 'POST'])
@cross_origin()
def train():
    train_data()
    return render_template('index.html')


@app.route('/predict',methods = ['GET', 'POST'])
@cross_origin()
def predict():

    if request.method == 'POST':
        try:
            # file_object = open("logs/GeneralLogs.txt", 'a+')
            # log_writer.log(file_object, 'Start getting data from UI')
            age = float(request.form['age'])
            workclass = (request.form['workclass'])
            fnlwgt = float(request.form['fnlwgt'])
            education = (request.form['education'])
            education_num = float(request.form['education_num'])
            marital_status = (request.form['marital_status'])
            occupation = (request.form['occupation'])
            relationship = (request.form['relationship'])
            race = (request.form['race'])
            sex = (request.form['sex'])
            capital_gain = float(request.form['capital_gain'])
            capital_loss = float(request.form['capital_loss'])
            hours_per_week = float(request.form['hours_per_week'])
            native_country = (request.form['native_country'])


            # log_writer.log(file_object, 'Complete getting data from UI')

            mydict = {'age':age,	'workclass':workclass,	'fnlwgt':fnlwgt,	'education':education,	'education_num':education_num,
              'marital_status':marital_status,	'occupation':occupation,	'relationship':relationship,	'race':race,
              'sex':sex, 'capital_gain':capital_gain,	'capital_loss':capital_loss,	'hours_per_week':hours_per_week,
              'native_country':native_country}
            # log_writer.log(file_object, 'Passing mydict to prediction.predict_data')
            prediction = predict_data(mydict)
            if prediction == 0:
                result = '<=50K'
            else:
                result = '>50K'
            return render_template('results.html', result=result)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
            # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__=='__main__':
    app.run(debug=True, host='127.0.0.1', port =5001 )