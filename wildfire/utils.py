import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pylab import *
from scipy.interpolate import interp2d


def cal_accuracy(prediction, label):
    return accuracy_score(prediction, label)


def get_accuracy_from_posterior(mean, testing_label):
    mean = np.squeeze(mean)
    prediction = np.ones_like(testing_label)
    prediction[np.where(mean <= 0.5)] = 0
    return cal_accuracy(prediction, testing_label)


def get_tpr(prediction, label):
    cm = confusion_matrix(label, prediction)
    return cm[1][1] / np.sum(cm[1])


def get_tpr_from_posterior(mean, testing_label):
    mean = np.squeeze(mean)
    prediction = np.ones_like(testing_label)
    prediction[np.where(mean <= 0.5)] = 0
    return get_tpr(prediction, testing_label)


def plot_the_distribution(data, loc, true_label):
    cmap = plt.get_cmap('PiYG')

    fig = plt.figure()
    plt.suptitle('Prediction variance', fontsize=16)

    ax1 = fig.add_subplot(121)
    ax1.set_title('positive')
    indices = np.where(true_label == 1)
    data1 = data[indices]
    loc1 = loc[indices]
    x = loc1[:, 1]
    y = loc1[:, 0]
    cax = ax1.scatter(x, y, c=data1, marker='.', s=100, cmap=cmap, vmin=data1.min(), vmax=data1.max())
    fig.colorbar(cax, extend='min')

    ax2 = fig.add_subplot(122)
    ax2.set_title('negative')
    indices = np.where(true_label == 0)
    data2 = data[indices]
    loc2 = loc[indices]
    x = loc2[:, 1]
    y = loc2[:, 0]
    cax = ax2.scatter(x, y, c=data2, marker='.', s=100, cmap=cmap, vmin=data2.min(), vmax=data2.max())
    fig.colorbar(cax, extend='min')

    plt.show()

