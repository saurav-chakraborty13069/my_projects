from wsgiref import simple_server
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
from logistic_deploy import predObj

app = Flask(__name__)
# CORS(app)
# app.config['DEBUG'] = True


# class ClientApi:
#
#     def __init__(self):
#         self.predObj = predObj()

# @app.route("/predict", methods=['POST'])
# def predictRoute():
#     try:
#         if request.json['data'] is not None:
#             data = request.json['data']
#             print('data is:     ', data)
#             pred=predObj()
#             res = pred.predict_log(data)
#
#             #result = clntApp.predObj.predict_log(data)
#             print('result is        ',res)
#             return Response(res)
#     except ValueError:
#         return Response("Value not found")
#     except Exception as e:
#         print('exception is   ',e)
#         return Response(e)



@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            pregnancies=int(request.form['pregnancies'])
            glucose = float(request.form['glucose'])
            bp = float(request.form['blood_pressure'])
            thickness = float(request.form['skin_thickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['diabetes_function'])
            age = int(request.form['age'])
            # is_research = request.form['research']
            # if(is_research=='yes'):
            #     research=1
            # else:
            #     research=0
            filename = 'modelForPrediction.sav'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            scaler = pickle.load(open('sandardScalar.sav', 'rb'))
            mydict = {'Pregnancies': pregnancies, 'Glucose': glucose, 'blood_pressure': bp,
                      'skin_thickness': thickness, 'insulin': insulin, 'BMI':bmi, 'diabetes_function': dpf, 'Age': age}
            data_df = pd.DataFrame(mydict,index = [1,])
            print(data_df)
            # predictions using the loaded model file
            prediction=loaded_model.predict(scaler.transform(data_df))
            print('prediction is', prediction)
            if prediction[0] == 1:
                result = 'Diabetic'
            else:
                result = 'Non-Diabetic'
            # showing the prediction results in a UI

            return render_template('results.html', prediction=result)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    # clntApp = ClientApi()
    app.run(host='127.0.0.1', port=8001, debug=True)
    # host = '0.0.0.0'
    # port = 5000
    # app.run(debug=True)
    #httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    #httpd.serve_forever()