import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import geopandas as gpd


def fire_data_with_geopandas(filename):
    fire_map = gpd.read_file(filename)
    centers = fire_map.geometry.centroid
    # for row in fire_map:
    #     print(row)
    # centers.plot()
    # plt.show()
    return fire_map, centers


def fire_data_with_basemap(filename):
    fire_map = Basemap(llcrnrlon=-125, llcrnrlat=32, urcrnrlon=-115, urcrnrlat=43)
    fire_map.readshapefile(filename, 'fires')
    fire_centers = list(map(lambda x: np.mean(x, axis=0), fire_map.fires))
    plt.figure()
    x = list(map(lambda a: a[0], fire_centers))
    y = list(map(lambda a: a[1], fire_centers))
    plt.scatter(x, y, marker='.')
    plt.show()
    return fire_map, fire_centers


def read_features(filename):
    with open(filename, 'r') as fi:
        data = fi.readlines()
    data = data[0].split('}},')
    for itr, line in enumerate(data):
        data[itr] = line.split(',')
    return data


def get_features(indices, features, feature_names):
    output_data = []
    for itr, index in enumerate(indices):
        this_index_data = []
        data = features[index]
        for feature_name in feature_names:
            for line in data:
                if line.find(feature_name) != -1:
                    line = line.split(':')[-1].strip(' \'')
                    this_index_data.append(float(line))
                    break
        if len(this_index_data) == 9:
            output_data.append(this_index_data)
    return np.array(output_data)


def construct_feature_matrices(fire_map, features_file, year_start, year_end, year_predict):
    """
    Each row in the feature matrix corresponds to one fire even.
    Columns in feature matrix:
    0: X coordinate
    1: Y coordinate
    2: wind speed
    3: temperature high
    4: temperature low
    5: humidity
    6: pressure
    7: clod cover
    8: precipitation intensity
    """

    feature_names = ['latitude', 'longitude', 'windSpeed', 'temperatureHigh', 'temperatureLow',
                     'humidity', 'pressure', 'cloudCover', 'precipIntensity']
    features = read_features(features_file)

    training_years = [str(year) for year in range(year_start, year_end + 1)]
    testing_years = [str(year_predict)]

    training_fire_map = fire_map.loc[fire_map['YEAR_'].isin(training_years)]
    testing_fire_map = fire_map.loc[fire_map['YEAR_'].isin(testing_years)]

    # construct the training matrix
    training_data = get_features(training_fire_map.index.tolist(), features, feature_names)
    testing_data = get_features(testing_fire_map.index.tolist(), features, feature_names)

    return training_data, testing_data


def construct_negative_examples(f1, f2):
    feature_names = ['latitude', 'longitude', 'windSpeed', 'temperatureHigh', 'temperatureLow',
                     'humidity', 'pressure', 'cloudCover', 'precipIntensity']
    training_features = read_features(f1)
    testing_features = read_features(f2)

    training_indices = [i for i in range(len(training_features))]
    testing_indices = [i for i in range(len(testing_features))]

    training_data = get_features(training_indices, training_features, feature_names)
    testing_data = get_features(testing_indices, testing_features, feature_names)

    return training_data, testing_data


def get_data(fire_map_f, positive_f, negative_training_f, negative_testing_f, plot_fig=False):
    fire_map, _ = fire_data_with_geopandas(fire_map_f)

    training_p, testing_p = construct_feature_matrices(fire_map, positive_f,
                                                       year_start=2000, year_end=2015, year_predict=2016)
    training_n, testing_n = construct_negative_examples(negative_training_f, negative_testing_f)

    training_features = np.concatenate((training_p, training_n), axis=0)
    training_label = np.append(np.ones(training_p.shape[0]), -1*np.ones(training_n.shape[0]))
    testing_features = np.concatenate((testing_p, testing_n), axis=0)
    testing_label = np.append(np.ones(testing_p.shape[0]), -1*np.ones(testing_n.shape[0]))

    if plot_fig:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.set_title('positive')
        ax1.scatter(training_p[:, 1], training_p[:, 0], marker='.', color='#0172B2')
        ax1.scatter(testing_p[:, 1], testing_p[:, 0], marker='.', color='#CC6600')
        ax2 = fig.add_subplot(122)
        ax2.set_title('negative')
        ax2.scatter(training_n[:, 1], training_n[:, 0], marker='.', color='#0172B2')
        ax2.scatter(testing_n[:, 1], testing_n[:, 0], marker='.', color='#CC6600')
        plt.show()

    return training_features, training_label, testing_features, testing_label


def main():
    fire_map_f = '../data/fire16/fires_latlong_16_1.shp'
    positive_f = '../data/darksky.txt'
    negative_training_f = '../data/darksky_none-fire.txt'
    negative_testing_f = '../data/darksky_none-2016-fire.txt'
    training_features, training_label, testing_features, testing_label = \
        get_data(fire_map_f, positive_f, negative_training_f, negative_testing_f, plot_fig=True)
    np.save('../data/data.npy', [training_features, training_label, testing_features, testing_label])

    return 0


if __name__ == "__main__":
    main()
