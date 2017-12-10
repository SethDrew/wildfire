import sys  
sys.path.append('lib')


import requests
import cfg
import numpy as np
import pandas as pd
import geopandas as gpd
import json

m=gpd.read_file('data/latlong_fires/fires_latlong_16_1.shp')
times = pd.to_datetime(m.ALARM_DATE)
dates = []
centers = m.geometry.centroid

for (point, time, obj_id) in zip(centers, m.ALARM_DATE, m.OBJECTID):

    lat = point.y
    lon = point.x
    req = 'https://api.darksky.net/forecast/{key}/{lat},{lon},{time}T12:00:00?exclude=hourly,minutely,currently,alerts,flags'.format(lat=lat,lon=lon,key=cfg.DS_KEY, time=time)
    # dates.append({obj_id:requests.get(req).json()})


with open("darksky.txt", 'w') as f:
    json.dump(dates, f)
