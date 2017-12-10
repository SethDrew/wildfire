import numpy as np
import gpflow
import tensorflow as tf
from gpflow.decors import params_as_tensors
import utils


class Customize_Linear(gpflow.mean_functions.MeanFunction):
    def __init__(self, A=None, b=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.A = gpflow.params.Parameter(np.atleast_2d(A))
        self.b = gpflow.params.Parameter(b)

    @params_as_tensors
    def __call__(self, X):
        transform_m = tf.cast(tf.reshape(tf.constant([0.0, 0, 1, 1, 1, 1, 1, 1, 1]), [1, -1]), dtype=tf.float64)
        new_A = tf.reshape(tf.diag_part(tf.matmul(self.A, transform_m)), tf.shape(self.A))
        return tf.matmul(X, new_A) + self.b


def classify_GP_wo_loc(training_features, training_label, testing_features, testing_label, num_inducing, input_dim):
    Z = training_features[np.random.choice(len(training_features), num_inducing)]
    #
    # m = gpflow.models.SVGP(training_features, training_label.reshape(-1, 1),
    #                        kern=gpflow.kernels.RBF(input_dim=7),
    #                        likelihood=gpflow.likelihoods.Bernoulli(), Z=Z,
    #                        minibatch_size=10)

    # m = gpflow.models.VGP(training_features, training_label.reshape(-1, 1),
    #                       kern=gpflow.kernels.RBF(input_dim=7),
    #                       likelihood=gpflow.likelihoods.Bernoulli())

    # m = gpflow.models.GPMC(training_features, training_label.reshape(-1, 1),
    #                        kern=gpflow.kernels.RBF(input_dim=7),
    #                        likelihood=gpflow.likelihoods.Bernoulli())

    m = gpflow.models.SGPMC(training_features, training_label.reshape(-1, 1),
                            kern=gpflow.kernels.RBF(input_dim=input_dim),
                            likelihood=gpflow.likelihoods.Bernoulli(), Z=Z)

    m.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m, maxiter=10000)

    mean, var = m.predict_y(testing_features)
    np.save('../data/result_wo_loc.npy', [mean, var])

    return utils.get_accuracy_from_posterior(mean, testing_label)


def classify_GP_loc(training_features, training_label, testing_features, testing_label, num_inducing):
    _, m = training_features.shape
    linear_mean = Customize_Linear(np.random.random((m, 1)).astype(float), 0.0)

    Z = training_features[np.random.choice(len(training_features), num_inducing)]
    # m = gpflow.models.SVGP(training_features, training_label.reshape(-1, 1),
    #                        kern=gpflow.kernels.RBF(input_dim=2, active_dims=[0, 1]),
    #                        likelihood=gpflow.likelihoods.Bernoulli(), Z=Z, mean_function=linear_mean,
    #                        minibatch_size=10)

    # m = gpflow.models.GPMC(training_features, training_label.reshape(-1, 1),
    #                        kern=gpflow.kernels.RBF(input_dim=2, active_dims=[0, 1]),
    #                        likelihood=gpflow.likelihoods.Bernoulli(), mean_function=linear_mean)

    m = gpflow.models.SGPMC(training_features, training_label.reshape(-1, 1),
                            kern=gpflow.kernels.RBF(input_dim=2, active_dims=[0, 1]) +
                                 gpflow.kernels.RBF(input_dim=7, active_dims=[2, 3, 4, 5, 6, 7, 8]),
                            likelihood=gpflow.likelihoods.Bernoulli(), Z=Z, mean_function=linear_mean)

    m.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m, maxiter=10000)

    mean, var = m.predict_y(testing_features)
    np.save('../data/result_wo_loc.npy', [mean, var])

    return utils.get_accuracy_from_posterior(mean, testing_label)
