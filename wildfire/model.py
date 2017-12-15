import numpy as np
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import GP_gpflow
import utils
import tensorflow as tf


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

    for j in range(2):
        training_loc[:, j] = training_loc[:, j] - np.mean(training_loc[:, j])
        testing_loc[:, j] = testing_loc[:, j] - np.mean(testing_loc[:, j])

    return training_loc, training_features_wo_loc, testing_loc, testing_features_wo_loc


def read_data(file_path='../data/data.npy', location_info=False):
    training_features, training_label, testing_features, testing_label = np.load(file_path)

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

    print('accuracy:',utils.cal_accuracy(prediction, testing_label),
          'TPR:', utils.get_tpr(prediction, testing_label))

    return utils.cal_accuracy(prediction, testing_label)


def classify_SVM(training_features, training_label, testing_features, testing_label):
    clf = SVC()
    clf.fit(training_features, training_label)
    prediction = clf.predict(testing_features)

    print('accuracy:',utils.cal_accuracy(prediction, testing_label),
          'TPR:', utils.get_tpr(prediction, testing_label))

    return utils.cal_accuracy(prediction, testing_label)


def classify_random_forest(training_features, training_label, testing_features, testing_label):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(training_features, training_label)
    prediction = clf.predict(testing_features)

    print('accuracy:',utils.cal_accuracy(prediction, testing_label),
          'TPR:', utils.get_tpr(prediction, testing_label))

    return utils.cal_accuracy(prediction, testing_label)


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_uniform([in_size, out_size], minval=-1, maxval=1))
    biases = tf.Variable(tf.random_uniform([1, out_size], minval=0, maxval=0.1))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def neural_network(training_features, training_label, testing_features, testing_label):

    feature_n = training_features.shape[1]
    width_h1 = 5
    width_h2 = 3

    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, feature_n])
    ys = tf.placeholder(tf.float32, [None])

    l1 = add_layer(xs, feature_n, width_h1, activation_function=tf.nn.sigmoid)
    l2 = add_layer(l1, width_h1, width_h2, activation_function=tf.nn.sigmoid)
    prediction = add_layer(l2, width_h2, 1, activation_function=tf.nn.sigmoid)

    # the error between prediction and real data
    loss = tf.reduce_sum(tf.square(ys - tf.reshape(prediction, [-1])))

    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            _, losses = sess.run([train_step, loss], feed_dict={xs: training_features, ys: training_label})
        predict_labels = sess.run(tf.round(prediction), feed_dict={xs: testing_features, ys: testing_label})

    print('accuracy:',utils.cal_accuracy(predict_labels, testing_label),
          'TPR:', utils.get_tpr(predict_labels, testing_label))

    return utils.cal_accuracy(predict_labels, testing_label)


def main():
    num_inducing = 100

    # print('=======================without location==========================================')
    # print('=======================without location==========================================')
    # training_features, training_label, testing_features, testing_label = read_data(location_info=False)
    #
    # print('Neural network:', end='')
    # neural_network(training_features, training_label, testing_features, testing_label)
    #
    # print('Linear classification:', end='')
    # classify_linear_model(training_features, training_label, testing_features, testing_label)
    #
    # print('SVM:', end='')
    # classify_SVM(training_features, training_label, testing_features, testing_label)
    #
    # print('Random forest:', end='')
    # classify_random_forest(training_features, training_label, testing_features, testing_label)
    #
    # print('GP:', end='')
    # GP_gpflow.classify_GP_wo_loc(training_features, training_label, testing_features, testing_label,
    #                              num_inducing=num_inducing, input_dim=7)

    print('\n')
    print('=======================with location==========================================')
    print('=======================with location==========================================')

    training_features, training_label, testing_features, testing_label = read_data(location_info=True)
    #
    # print('Neural network:', end='')
    # neural_network(training_features, training_label, testing_features, testing_label)
    #
    # print('Linear classification:', end='')
    # classify_linear_model(training_features, training_label, testing_features, testing_label)
    #
    # print('SVM:', end='')
    # classify_SVM(training_features, training_label, testing_features, testing_label)
    #
    # print('Random forest:', end='')
    # classify_random_forest(training_features, training_label, testing_features, testing_label)

    # print('GP', end='')
    # GP_gpflow.classify_GP_wo_loc(training_features, training_label, testing_features, testing_label,
    #                              num_inducing=num_inducing, input_dim=9)

    print('GP (mean function: linear function and features, kernel: location):', end='')
    GP_gpflow.classify_GP_loc(training_features, training_label, testing_features, testing_label,
                              num_inducing=num_inducing, mean_fun='linear')

    # print('GP (mean function: neural network and features, kernel: location):', end='')
    # GP_gpflow.classify_GP_loc(training_features, training_label, testing_features, testing_label,
    #                           num_inducing=num_inducing, mean_fun='neural_net')

    return 0


def draw_dis():
    mean, var = np.load('../data/result.npy')
    mean = np.squeeze(mean)
    var = np.squeeze(var)
    training_features, training_label, testing_features, testing_label = np.load('../data/data.npy')
    loc = testing_features[:, :2]
    utils.plot_the_distribution(var, loc, testing_label)


def regression_areas():
    training_data, training_label, testing_data, testing_label = read_data(file_path='../data/areas.npy',
                                                                           location_info=True)

    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    import gpflow
    import tensorflow as tf

    # drop 10% head and 10% tail
    # N_remove = int(0.1*training_data.shape[0])
    # indices = np.argsort(training_label)[N_remove:-N_remove]
    # training_data = training_data[indices]
    # training_label = training_label[indices]
    #
    # N_remove = int(0.1*testing_data.shape[0])
    # indices = np.argsort(testing_label)[N_remove:-N_remove]
    # testing_data = testing_data[indices]
    # testing_label = testing_label[indices]

    # linear regression
    clf = linear_model.LinearRegression()
    clf.fit(training_data, training_label)
    prediction = clf.predict(testing_data)
    print('LR:', mean_squared_error(prediction, testing_label), clf.score(testing_data, testing_label))

    # nerual network
    feature_n = training_data.shape[1]
    width_h1 = 5
    width_h2 = 3
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, feature_n])
    ys = tf.placeholder(tf.float32, [None])
    l1 = add_layer(xs, feature_n, width_h1, activation_function=tf.nn.sigmoid)
    l2 = add_layer(l1, width_h1, width_h2, activation_function=tf.nn.sigmoid)
    prediction = add_layer(l2, width_h2, 1, activation_function=tf.nn.relu)
    # the error between prediction and real data
    loss = tf.reduce_sum(tf.square(ys - tf.reshape(prediction, [-1])))
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            _, losses = sess.run([train_step, loss], feed_dict={xs: training_data, ys: training_label})
        prediction = sess.run(prediction, feed_dict={xs: testing_data, ys: testing_label})
    print('NN:', mean_squared_error(prediction, testing_label))

    # SVM
    clf = SVR()
    clf.fit(training_data, training_label)
    prediction = clf.predict(testing_data)
    print('SVM:', mean_squared_error(prediction, testing_label))

    # GP
    Z = training_data[np.random.choice(len(training_data), 100)]
    m = gpflow.models.SGPMC(training_data, training_label.reshape(-1, 1),
                            kern=gpflow.kernels.RBF(input_dim=9),
                            likelihood=gpflow.likelihoods.Bernoulli(), Z=Z)
    m.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m, maxiter=10000)
    mean, var = m.predict_y(testing_data)
    print('GP:', mean_squared_error(np.squeeze(mean), testing_label))

    return 0


if __name__ == "__main__":
    main()
    # draw_dis()
    # regression_areas()
