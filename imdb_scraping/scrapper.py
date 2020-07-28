import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from random import randint
from time import time
from logger import App_Logger
from warnings import warn
import logger
from datetime import  datetime


def scrape(y, log_writer, file_object):
    #log_writer = logger.App_Logger()
    #file_object = open("logs/imdb_scraper_test-{}.txt".format(datetime.now().date()), 'a+')
    pages = [str(i) for i in range(1,100, 50)]
    #years_url = [str(i) for i in range(2017,2018)]

    movies = []

    headers = {"Accept-Language": "en-US, en;q=0.5"}
    start_time = time()
    request = 0

    # url = 'http://www.imdb.com/search/title?release_date=2017&sort=num_votes,desc&page=1'
    # response = requests.get(url)
    # #print(response.text[:500])
    # html_soup = BeautifulSoup(response.text, 'html.parser')
    # movie_containers = html_soup.find_all('div', class_ = 'lister-item mode-advanced')


    for page in pages:
        #url = 'http://www.imdb.com/search/title?release_date={}&sort=num_votes,desc&page={}'.format(y, page)
        url = 'https://www.imdb.com/search/title/?release_date={}-01-01,{}-12-31&sort=num_votes,desc&start={}&ref_=adv_nxt'.format(y,y, page)
        log_writer.log(file_object, 'reading from the url {}'.format(url))
        response = requests.get(url, headers = headers)
        log_writer.log(file_object, 'received the response')
        sleep(randint(8, 15))

        request += 1
        elapsed_time = time() - start_time
        log_writer.log(file_object, 'Request:{}; Frequency: {} requests/s'.format(request, request / elapsed_time))
        page_html = BeautifulSoup(response.text, 'html.parser')
        mv_containers = page_html.find_all('div', class_='lister-item mode-advanced')
        #print(mv_containers[0])
        log_writer.log(file_object, 'Start of getting tag details for page {}'.format(page))
        for container in mv_containers:
        #if container.find('div', class_ = 'ratings-metascore') is not None:
            try:
                name = container.h3.a.text
            except:
                name = 'No Name'

            try:
                year = container.h3.find('span',class_ = 'lister-item-year text-muted unbold').text
            except:
                year = 'No year'

            try:
                rating = container.strong.text
            except:
                rating = 'No Rating'

            try:
                metascore = container.find('span', class_ = 'metascore favorable').text
            except:
                metascore = "No metascore"

            try:
                vote = container.find('span', attrs = {'name': "nv"}).text
            except:
                vote = "no vote"

            try:
                desc = container.find_all('p', class_='text-muted')[1].text
            except:
                desc = 'No Description'

            try:
                cert = container.find('span', class_='certificate').text
            except:
                cert = 'No cert'

            try:
                runtime = container.find('span', class_='runtime').text
            except:
                runtime = 'No runtime'

            try:
                genre = container.find('span', class_='genre').text
            except:
                genre = 'No genre'

            try:
                director = container.find_all('p')[2].a.text
            except:
                director = 'No director'

            try:
                st = container.find_all('p')[2].find_all('a')[1:]
                star = [s.text for s in st]
            except:
                star = 'No Star'

            try:
                gross = container.find_all('span', attrs = {'name': "nv"})[1].text
            except:
                gross = 'No Gross'

            mydict = {'Name': name, 'Year': year, 'Rating': rating, 'Metascore': metascore,
                      'Vote': vote, 'Description': desc, 'Certificate': cert, 'Runtime': runtime,
                      'Genre': genre, 'Director': director,'Stars': star, 'Earning': gross}
            movies.append(mydict)

        log_writer.log(file_object, 'End of getting tag details for page {}'.format(page))

    log_writer.log(file_object, 'returning movies')
    return movies






