import numpy as np


class Optimizer:
    """
    Optimizer class
    """

    def __init__(self, net, cost, learning_rate, *args, **kwargs):
        self.net = net
        self.cost = cost
        self.learning_rate = learning_rate

    def compute_gradients(self, batch, y, *args, **kwargs):
        zs, as_ = self.net.forward_pass(batch)
        gradients = []
        m = y.shape[0]
        dA = self.cost.prime(as_[-1], y.T)
        for i in range(len(self.net.weights) - 1, 0, -1):
            dZ = dA * self.net.activation_prime(zs[i])
            dW = np.matmul(dZ, as_[i - 1].T) / m
            gradients = [dW] + gradients
            dA = np.matmul(self.net.weights[i].T, dZ)
        dZ = dA * self.net.activation_prime(zs[0])
        dW = np.matmul(dZ, batch) / m
        gradients = [dW] + gradients
        return gradients

    def update_weights(self, *args, **kwargs):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimizer
    """

    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

    def update_weights(self, batch):
        batch_xs, batch_ys = batch
        gradients = self.compute_gradients(batch_xs, batch_ys)

        for w, dW in zip(self.net.weights, gradients):
            w -= self.learning_rate * dW

    def name(self):
        return 'SGD'


class MomentumOptimizer(Optimizer):
    """
    SGD With Momentum Optimizer
    """

    def __init__(self, *args, gamma=0.9, **kwargs):
        super(MomentumOptimizer, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.past_gradients = []
        for w in self.net.weights:
            self.past_gradients.append(np.zeros_like(w))

    def update_weights(self, batch):
        batch_xs, batch_ys = batch
        gradients = self.compute_gradients(batch_xs, batch_ys)

        for i, dW in enumerate(gradients):
            # add momemtum term to weights update
            self.net.weights[i] -= self.gamma * self.past_gradients[i] + self.learning_rate * dW
            self.past_gradients[i] = dW

    def name(self):
        return 'Momentum'


class NAG(Optimizer):
    """
    Nesterov Accelerated Gradient Optimizer
    """

    def __init__(self, *args, gamma=0.9, **kwargs):
        super(NAG, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.past_gradients = []
        for w in self.net.weights:
            self.past_gradients.append(np.zeros_like(w))

    def compute_gradients(self, batch, y, *args, **kwargs):
        net_weights = []
        for w in self.net.weights:
            net_weights.append(np.copy(w))

        # compute gradients with respect to approximated future parameters
        for i, w in enumerate(self.net.weights):
            self.net.weights[i] = w - self.gamma * self.past_gradients[i]

        gradients = super(NAG, self).compute_gradients(batch, y)

        # restore weights
        self.net.weights = net_weights
        return gradients

    def update_weights(self, batch):
        batch_xs, batch_ys = batch
        gradients = self.compute_gradients(batch_xs, batch_ys)

        for i, dW in enumerate(gradients):
            # add momentum term
            self.net.weights[i] -= self.gamma * self.past_gradients[i] + self.learning_rate * dW
            self.past_gradients[i] = dW

    def name(self):
        return 'NAG'


class Adagrad(Optimizer):
    """
    Adagrad Optimizer
    """

    def __init__(self, *args, epsilon=1e-8, **kwargs):
        super(Adagrad, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.gradient_squares = []
        for w in self.net.weights:
            self.gradient_squares.append(np.zeros_like(w))

    def update_weights(self, batch):
        batch_xs, batch_ys = batch
        gradients = self.compute_gradients(batch_xs, batch_ys)

        for i, dW in enumerate(gradients):
            # accumulate gradients squares since the beginning
            self.gradient_squares[i] += np.square(dW)
            self.net.weights[i] -= self.learning_rate / (np.sqrt(self.gradient_squares[i] + self.epsilon)) * dW

    def name(self):
        return 'Adagrad'


class Adadelta(Optimizer):
    """
    Adadelta Optimizer
    """

    def __init__(self, *args, gamma=0.9, epsilon=1e-8, **kwargs):
        super(Adadelta, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.gradients_squares = []
        self.past_updates_squares = []
        for w in self.net.weights:
            self.gradients_squares.append(np.zeros_like(w))
            self.past_updates_squares.append(np.zeros_like(w))

    def update_weights(self, batch):
        batch_xs, batch_ys = batch
        gradients = self.compute_gradients(batch_xs, batch_ys)

        for i, dW in enumerate(gradients):
            # decay accumulated gradients squares
            self.gradients_squares[i] = self.gamma * self.gradients_squares[i] + (1 - self.gamma) * dW ** 2
            update = -np.sqrt(
                (self.past_updates_squares[i] + self.epsilon) / (self.gradients_squares[i] + self.epsilon)) * dW
            self.past_updates_squares[i] = np.square(
                self.gamma * self.past_updates_squares[i] + (1 - self.gamma) * update)
            self.net.weights[i] += update

    def name(self):
        return 'Adadelta'


class RMSProp(Optimizer):
    """
    RMSProp Optimizer
    """

    def __init__(self, *args, gamma=0.9, epsilon=1e-8, **kwargs):
        super(RMSProp, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.gradients_squares = []
        for w in self.net.weights:
            self.gradients_squares.append(np.zeros_like(w))

    def update_weights(self, batch):
        batch_xs, batch_ys = batch
        gradients = self.compute_gradients(batch_xs, batch_ys)

        for i, dW in enumerate(gradients):
            # decay accumulated gradients squares
            self.gradients_squares[i] = self.gamma * self.gradients_squares[i] + (1 - self.gamma) * dW ** 2
            update = -self.learning_rate / np.sqrt(self.gradients_squares[i] + self.epsilon) * dW
            self.net.weights[i] += update

    def name(self):
        return 'RMSProp'


class Adam(Optimizer):
    """
    Adam Optimizer
    """

    def __init__(self, *args, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step = 1
        self.past_gradients = []
        self.gradient_squares = []
        for w in self.net.weights:
            self.past_gradients.append(np.zeros_like(w))
            self.gradient_squares.append(np.zeros_like(w))

    def update_weights(self, batch):
        batch_xs, batch_ys = batch
        gradients = self.compute_gradients(batch_xs, batch_ys)

        for i, dW in enumerate(gradients):
            # decay accumulated gradients
            self.past_gradients[i] = self.beta1 * self.past_gradients[i] + (1 - self.beta1) * dW
            # decay accumulated gradients squares
            self.gradient_squares[i] = self.beta2 * self.gradient_squares[i] + (1 - self.beta2) * dW ** 2
            # compute corrected estimates
            mean_estimate = self.past_gradients[i] / (1 - self.beta1 ** self.step)
            var_estimate = self.gradient_squares[i] / (1 - self.beta2 ** self.step)
            update = -self.learning_rate / (np.sqrt(var_estimate) + self.epsilon) * mean_estimate
            self.net.weights[i] += update

        self.step += 1

    def name(self):
        return 'Adam'
