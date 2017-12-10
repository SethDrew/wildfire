import numpy as np
from sklearn.metrics import accuracy_score


def cal_accuracy(prediction, label):
    return accuracy_score(prediction, label)


def get_accuracy_from_posterior(mean, testing_label):
    mean = np.squeeze(mean)
    prediction = np.ones_like(testing_label)
    prediction[np.where(mean < 0.5)] = -1.0
    return cal_accuracy(prediction, testing_label)

