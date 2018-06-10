import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from optimizers import SGD, MomentumOptimizer, NAG, Adagrad, Adadelta, RMSProp, Adam

MNIST_IMG_H = 28
MNIST_IMG_W = 28

BATCH_SIZE = 512

EPSILON = 1e-8

NUM_STEPS = 5000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class CostFn:
    def cost(self, logits, y, *args, **kwargs):
        """
        Compute cost
        :param logits: logits
        :param y: ground truth
        :return: cost
        """
        raise NotImplementedError

    def prime(self, logits, y, *args, **kwargs):
        """
        Cost prime for backpropagation
        :param logits: logits
        :param y: ground truth
        :return:
        """
        raise NotImplementedError


class CrossEntropyLoss(CostFn):
    """
    Cross entropy loss function
    """

    def cost(self, logits, y, *args, **kwargs):
        return np.mean(np.nan_to_num(-y * np.log(logits) - (1 - y) * np.log(1 - logits)))

    def prime(self, logits, y, *args, **kwargs):
        return logits - y


class MLP:
    def __init__(self, num_neurons, activation=sigmoid, activation_prime=sigmoid_prime):
        self.num_layers = len(num_neurons)
        self.num_neurons = num_neurons
        self.activation = activation
        self.activation_prime = activation_prime

        self.weights = []
        for i, n in enumerate(self.num_neurons[1:], 1):
            w = np.random.randn(n, num_neurons[i - 1])
            self.weights.append(w)

    def forward_pass(self, batch):
        inputs = batch.T
        zs = []
        as_ = []
        for W in self.weights:
            z = np.matmul(W, inputs)
            zs.append(z)
            a = self.activation(z)
            as_.append(a)
            inputs = a
        return zs, as_

    def inference(self, batch):
        inputs = batch.T
        for W in self.weights:
            z = np.matmul(W, inputs)
            a = self.activation(z)
            inputs = a
        return inputs.T


def accuracy(logits, labels):
    cls = np.argmax(softmax(logits), axis=1)
    correct = np.argmax(labels, axis=1)
    acc = np.sum(cls == correct) / len(labels)
    return acc


def main():
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    for lr in [0.001, 0.01, 0.1, 1.0, 10.0]:
        _optimizers = [
            SGD,
            MomentumOptimizer,
            NAG,
            Adagrad,
            Adadelta,
            RMSProp,
            Adam,
        ]

        optimizers = []

        mlps = []
        for i in range(len(_optimizers)):
            np.random.seed(0)
            mlp = MLP([MNIST_IMG_H * MNIST_IMG_W, 30, 10])
            mlps.append(mlp)
            optimizers.append(_optimizers[i](net=mlp, cost=CrossEntropyLoss(), learning_rate=lr))

        errors = []
        for i in range(len(optimizers)):
            errors.append([])

        for step in range(NUM_STEPS):
            batch = mnist.train.next_batch(BATCH_SIZE)
            valid = mnist.validation.next_batch(BATCH_SIZE)
            if step % 10 == 0:
                print('step: {}'.format(step))
            for i, optimizer in enumerate(optimizers):
                optimizer.update_weights(batch)
                out = mlps[i].inference(valid[0])
                err = optimizer.cost.cost(out, valid[1])
                errors[i].append(err)
                if step % 10 == 0:
                    print('({})\terr: {:.8f}'.format(optimizers[i].name(), err))

        # make line a little more smooth
        for i, error in enumerate(errors):
            error_ = []
            for j, err in enumerate(error):
                error_.append(0.2 * err + 0.8 * error_[j-1] if j > 0 else err)
            errors[i] = error_

        fig, ax = plt.subplots(1, 1)
        for i, err in enumerate(errors):
            ax.plot(np.arange(0, NUM_STEPS, 1), err, label=optimizers[i].name())

        ax.set_title('Optimizers comparison [5000 steps, lr = {}]'.format(lr))
        ax.set_xlabel('Step')
        ax.set_ylabel('Error')
        ax.legend()
        fig.savefig('plots/plot_{}.png'.format(lr))


if __name__ == '__main__':
    main()
