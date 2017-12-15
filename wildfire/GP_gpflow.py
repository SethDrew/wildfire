import numpy as np
import gpflow
import tensorflow as tf
from gpflow.decors import params_as_tensors
import utils


class Customize_Linear(gpflow.mean_functions.MeanFunction):
    def __init__(self, A=None, b=None):
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.A = gpflow.params.Parameter(np.atleast_2d(A))
        self.b = gpflow.params.Parameter(b)

        tmp = np.zeros([9, 7])
        for i in range(7):
            tmp[i+2, i] = 1
        self.remove_loc = tf.constant(tmp, dtype=tf.float64)

    @params_as_tensors
    def __call__(self, X):
        return tf.matmul(tf.matmul(X, self.remove_loc), self.A) + self.b
        # return tf.matmul(X, self.A) + self.b


class Customize_NeuralNet(gpflow.mean_functions.MeanFunction):
    def __init__(self):
        gpflow.mean_functions.MeanFunction.__init__(self)

        self.width_h1 = 5
        self.width_h2 = 3

        self.Weights_1 = gpflow.params.Parameter(np.random.random([9, self.width_h1]))
        self.biases_1 = gpflow.params.Parameter(np.random.random([1, self.width_h1]))

        self.Weights_2 = gpflow.params.Parameter(np.random.random([self.width_h1, self.width_h2]))
        self.biases_2 = gpflow.params.Parameter(np.random.random([1, self.width_h2]))

        self.Weights_o = gpflow.params.Parameter(np.random.random([self.width_h2, 1]))
        self.biases_o = gpflow.params.Parameter(np.random.random([1, 1]))

    @params_as_tensors
    def __call__(self, X):
        f = X
        f = tf.nn.sigmoid(tf.matmul(f, self.Weights_1) + self.biases_1)
        f = tf.nn.sigmoid(tf.matmul(f, self.Weights_2) + self.biases_2)
        f = tf.nn.sigmoid(tf.matmul(f, self.Weights_o) + self.biases_o)
        return f


def classify_GP_wo_loc(training_features, training_label, testing_features, testing_label, num_inducing, input_dim):
    Z = training_features[np.random.choice(len(training_features), num_inducing)]

    m = gpflow.models.SGPMC(training_features, training_label.reshape(-1, 1),
                            kern=gpflow.kernels.RBF(input_dim=input_dim),
                            likelihood=gpflow.likelihoods.Bernoulli(), Z=Z)

    m.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m, maxiter=10000)

    mean, var = m.predict_y(testing_features)

    print('accuracy:',utils.get_accuracy_from_posterior(mean, testing_label),
          'TPR:', utils.get_tpr_from_posterior(mean, testing_label))

    return utils.get_accuracy_from_posterior(mean, testing_label)


def classify_GP_loc(training_features, training_label, testing_features, testing_label, num_inducing, mean_fun):
    Z = training_features[np.random.choice(len(training_features), num_inducing)]

    if mean_fun == 'linear':
        _, m = training_features.shape
        linear_mean = Customize_Linear(np.random.random((7, 1)).astype(float), 0.0)
        m = gpflow.models.SGPMC(training_features, training_label.reshape(-1, 1),
                                kern=gpflow.kernels.RBF(input_dim=2, active_dims=[0, 1]),
                                # kern=gpflow.kernels.RBF(input_dim=9),
                                likelihood=gpflow.likelihoods.Bernoulli(), Z=Z, mean_function=linear_mean)
    elif mean_fun == 'neural_net':
        neuralNet_mean = Customize_NeuralNet()
        m = gpflow.models.SGPMC(training_features, training_label.reshape(-1, 1),
                                kern=gpflow.kernels.RBF(input_dim=9),
                                likelihood=gpflow.likelihoods.Bernoulli(), Z=Z, mean_function=neuralNet_mean)
    else:
        raise RuntimeError

    m.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m, maxiter=10000)

    mean, var = m.predict_y(testing_features)

    np.save('../data/result.npy', [mean, var])

    print('accuracy:',utils.get_accuracy_from_posterior(mean, testing_label),
          'TPR:', utils.get_tpr_from_posterior(mean, testing_label))

    return utils.get_accuracy_from_posterior(mean, testing_label)
