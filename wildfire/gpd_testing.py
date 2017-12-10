import sys  
sys.path.append('lib')

import matplotlib.pyplot as plt
import geopandas as gpd



m =  gpd.read_file('data/latlong_fires/fires_latlong_16_1.shp')


calc_acres = m.geometry.to_crs({'init': 'epsg:3395'}).map(lambda p: p.area * 247.105/ 10**6)
gis_reported_acres = m.GIS_ACRES  # these two are similar but different
centers = m.geometry.centroid


