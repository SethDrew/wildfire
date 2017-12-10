import numpy as np
import pyGPs


def classify_GP(training_features, training_label, testing_features, testing_label):
    model = pyGPs.GPC()  # binary classification (default inference method: EP)
    # model.getPosterior(training_features, training_label)  # fit default model (mean zero & rbf kernel) with data
    model.optimize(training_features, training_label)  # optimize hyperparamters (default optimizer: single run minimize)
    a = model.predict(testing_features)  # predict test cases

    return a


def test():
    print(1)
