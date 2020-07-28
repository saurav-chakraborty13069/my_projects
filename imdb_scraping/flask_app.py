import os, shutil
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import requests
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen as uReq
import pandas as pd
from pymongo import MongoClient
import logger
import scrapper


app = Flask(__name__)

@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/review', methods=['POST', 'GET'])  # route to show the review comments in a web UI
@cross_origin()
def index():
    log_writer = logger.App_Logger()
    file_object = open("logs/imdb_scraper-{}.txt".format(datetime.now().date()), 'a+')
    if request.method == 'POST':
        log_writer.log(file_object, 'Getting the year')
        year = request.form['content']
        year = "".join(year.split())
        log_writer.log(file_object, 'received the year {}'.format(year))

        try:
            log_writer.log(file_object, 'connecting to mongo server')
            dbConn = MongoClient("mongodb://localhost:27017/")  # opening a connection to Mongo
            log_writer.log(file_object, 'connecting to db')
            db = dbConn['imdb_scrapper']  # connecting to the database called crawlerDB
            log_writer.log(file_object, 'creating/retrieving collection {}'.format(year))
            collection_name = 'movies_{}'.format(year)
            collection = db[collection_name]
            movies = db[collection_name].find({})  # searching the collection with the name same as the keyword
            if movies.count() > 0:
                log_writer.log(file_object, 'showing results from db')
                return render_template('results.html', movies=movies)
            else:
                log_writer.log(file_object, 'callign scrape function')
                movies = scrapper.scrape(year, log_writer, file_object)
                filename = 'movies_{}.csv'.format(year)

                try:
                    log_writer.log(file_object, 'creating dataframe and writing to CSV file')
                    df = pd.DataFrame(movies)
                    df.to_csv('./csv/{}'.format(filename))
                except Exception as e:

                    log_writer.log(file_object, "Exception occurred while creating csv file: {}".format(e))

                try:
                    files = os.listdir()
                    for f in files:
                        if f.endswith('.csv'):
                            shutil.move(f, 'csv')
                except Exception as e:
                    log_writer.log(file_object, "Exception occurred while moving csv file: {}".format(e))

                log_writer.log(file_object, 'inserting into collection {}'.format(year))
                collection.insert_many(df.to_dict('records'))

                return render_template('results.html', movies=movies[0:(len(movies) - 1)])

        except Exception as e:
            log_writer.log(file_object, "Exception occurred : {}".format(e))
            return 'something is wrong'
    else:
        return render_template('index.html')

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=5000)
    #app.run(host='0.0.0.0', port=port)
    app.run(host='127.0.0.1', port=8001, debug=True)

