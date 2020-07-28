import pandas as pd
import requests
from bs4 import BeautifulSoup


page = requests.get("http://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168")
soup = BeautifulSoup(page.content, 'html.parser')
seven_day = soup.find(id="seven-day-forecast-container")

periods = [pt.get_text() for pt in seven_day.select(".tombstone-container .period-name")]
short_descs = [sd.get_text() for sd in seven_day.select(".tombstone-container .short-desc")]
temps = [t.get_text() for t in seven_day.select(".tombstone-container .temp")]
descs = [d["title"] for d in seven_day.select(".tombstone-container img")]
# print(periods)
# print(short_descs)
# print(temps)
# print(descs)

weather = pd.DataFrame({
    "period": periods,
    "short_desc": short_descs,
    "temp": temps,
    "desc":descs
})

weather.to_csv('weather.csv')






