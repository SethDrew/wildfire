import numpy
import tensorflow as tf
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def read_data_with_basemap(filename):
    fire_map = Basemap(llcrnrlon=-125,llcrnrlat=23,urcrnrlon=-66,urcrnrlat=50)
    fire_map.readshapefile(filename, 'walls')

    for info, shape in zip(fire_map.walls_info, fire_map.walls):
        if info['COUNT_']:
            print(info['STATE'])

    plt.show()

def main():
    read_data_with_basemap('../data/Wildfires_US_2001-2009/GISPORTAL_GISOWNER01_USWILDFIRES0109')

if __name__ == "__main__":
    main()
