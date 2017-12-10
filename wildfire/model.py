import numpy as np
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt
import GP_gpflow
import GP_pyGPs
import utils
np.random.seed(0)


def data_preprocessing(training_features, testing_features):
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
    This function will separate the locations and normalize the features.
    """
    training_loc = training_features[:, :2]
    training_features_wo_loc = training_features[:, 2:]
    testing_loc = testing_features[:, :2]
    testing_features_wo_loc = testing_features[:, 2:]

    # normalize features
    _, m = training_features_wo_loc.shape
    for j in range(m):
        this_col = np.append(training_features_wo_loc[:, j], testing_features_wo_loc[:, j])
        min_val = min(this_col)
        max_val = max(this_col)
        training_features_wo_loc[:, j] = (training_features_wo_loc[:, j] - min_val) / (max_val - min_val)
        testing_features_wo_loc[:, j] = (testing_features_wo_loc[:, j] - min_val) / (max_val - min_val)

    return training_loc, training_features_wo_loc, testing_loc, testing_features_wo_loc


def read_data(location_info=False):
    training_features, training_label, testing_features, testing_label = np.load('../data/data.npy')

    training_loc, training_features_wo_loc, testing_loc, testing_features_wo_loc = \
        data_preprocessing(training_features, testing_features)

    if location_info:
        return np.concatenate((training_loc, training_features_wo_loc), axis=1), training_label, \
               np.concatenate((testing_loc, testing_features_wo_loc), axis=1), testing_label
    else:
        return training_features_wo_loc, training_label, testing_features_wo_loc, testing_label


def classify_linear_model(training_features, training_label, testing_features, testing_label):
    clf = linear_model.LogisticRegression()
    clf.fit(training_features, training_label)
    prediction = clf.predict(testing_features)
    return utils.cal_accuracy(prediction, testing_label)


def classify_SVM(training_features, training_label, testing_features, testing_label):
    clf = SVC()
    clf.fit(training_features, training_label)
    prediction = clf.predict(testing_features)
    return utils.cal_accuracy(prediction, testing_label)


def classify_random_forest(training_features, training_label, testing_features, testing_label):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(training_features, training_label)
    prediction = clf.predict(testing_features)
    return utils.cal_accuracy(prediction, testing_label)


def main():
    num_inducing = 100

    training_features, training_label, testing_features, testing_label = read_data(location_info=False)
    print('Linear classification (without location):', classify_linear_model(training_features, training_label, testing_features, testing_label))
    print('RBF SVM (without location):', classify_SVM(training_features, training_label, testing_features, testing_label))
    print('Random forest (without location):', classify_random_forest(training_features, training_label, testing_features, testing_label))
    print('GP without location:',
          GP_gpflow.classify_GP_wo_loc(training_features, training_label, testing_features, testing_label,
                                       num_inducing=num_inducing, input_dim=7))

    training_features, training_label, testing_features, testing_label = read_data(location_info=True)
    print('Linear classification (with location):', classify_linear_model(training_features, training_label, testing_features, testing_label))
    print('RBF SVM (with location):', classify_SVM(training_features, training_label, testing_features, testing_label))
    print('Random forest (with location):', classify_random_forest(training_features, training_label, testing_features, testing_label))
    print('GP with location (all features as kernel):',
          GP_gpflow.classify_GP_wo_loc(training_features, training_label, testing_features, testing_label,
                                       num_inducing=num_inducing, input_dim=9))
    print('GP with location (features as mean function, location as kernel):',
          GP_gpflow.classify_GP_loc(training_features, training_label, testing_features, testing_label,
                                    num_inducing=num_inducing))

    # a = []
    # for num_inducing in range(10,501,10):
    #     tmp = GP_gpflow.classify_GP_loc(training_features, training_label, testing_features, testing_label,
    #                                     num_inducing=num_inducing)
    #     print('GP with location (features as mean function, location as kernel) '+str(num_inducing)+':', tmp)
    #     a.append(tmp)
    # plt.plot(a)
    # plt.show()


    return 0


if __name__ == "__main__":
    main()
