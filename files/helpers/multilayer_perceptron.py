from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.utils import gen_batches, check_random_state
from sklearn.base import is_classifier

class UntrainedMLPClassifier(MLPClassifier):

    def _fit(self, X, y, incremental=False):
        print("_fit")
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        # Validate input parameters.
        self._validate_hyperparameters()
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." %
                             hidden_layer_sizes)

        X, y = self._validate_input(X, y, incremental)
        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = ([n_features] + hidden_layer_sizes +
                       [self.n_outputs_])

        # check random state
        self._random_state = check_random_state(self.random_state)

        if not hasattr(self, 'coefs_') or (not self.warm_start and not
                                           incremental):
            # First time training the model
            self._initialize(y, layer_units)

        # lbfgs does not support mini-batches
        if self.solver == 'lbfgs':
            batch_size = n_samples
        elif self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > n_samples:
                warnings.warn("Got `batch_size` less than 1 or larger than "
                              "sample size. It is going to be clipped")
            batch_size = np.clip(self.batch_size, 1, n_samples)

        # Initialize lists
        activations = [X]
        activations.extend(np.empty((batch_size, n_fan_out))
                           for n_fan_out in layer_units[1:])
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_,
                      n_fan_out_ in zip(layer_units[:-1],
                                        layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                           layer_units[1:]]

        return self

    def _initialize(self, y, layer_units):
        # set all attributes, allocate weights etc for first call
        # Initialize parameters
        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = 3
        print("_initialize")

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Output for regression
        if not is_classifier(self):
            self.out_activation_ = 'identity'
        # Output for multi class
        elif self._label_binarizer.y_type_ == 'multiclass':
            self.out_activation_ = 'softmax'
        # Output for binary class and multi-label
        else:
            self.out_activation_ = 'logistic'

    def predict(self, X):
        y_pred = self._predict(X)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        return self._label_binarizer.inverse_transform(y_pred)

    def prepare(self, X, y, coefficients, thresholds):
        self._fit(X,y)
        self.coefs_ = []
        self.intercepts_ = []

        self.coefs_ = coefficients
        self.intercepts_ = thresholds