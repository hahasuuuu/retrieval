import os
import sys
import numpy as np
import pandas as pd
import timeit

def relu(x):
    """
    ReLU Activation
    :param x: input data
    :return: 0 if value <= 0, else x
    """
    x[x <= 0] = 0
    return x

def relu_derivatives(x):
    """
    Derivative of ReLU Activation
    :param x: input data
    :return: 0 if value <= 0, else 1
    """
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def softmax(x):
    """
    Softmax
    :param x: input data
    :return: multinomial distribution
    """
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class ToyModel(object):
    def __init__(self, layers):
        """
        Xavier Initialization of weights and biases
        :param layers:
        :return: Initialized weights and biases
        """
        self.named_parameters = {}  # model parameters
        self.activations = {}   # model activations
        self.layer_num = len(layers) - 1   # model layer num
        self.params = []    # model params

        # Random seed
        rand_state = np.random.RandomState(42)

        for i in range(self.layer_num):
            bound = np.sqrt(6. / (layers[i] + layers[i + 1]))
            self.named_parameters['W_{}'.format(i)] = rand_state.uniform(-bound, bound, (layers[i], layers[i + 1]))
            self.named_parameters['B_{}'.format(i)] = rand_state.uniform(-bound, bound, layers[i + 1])

        for param in ["W", "B"]:
            for i in range(self.layer_num):
                self.params.append(self.named_parameters['{}_{}'.format(param, i)])

    def parameters(self):
        return self.params

    def __call__(self, input):
        """
        forward function:
        :param input
        :return last activations
        """
        self.activations = {}
        self.activations["A"] = [input]
        self.activations["Z"] = []
        for i in range(self.layer_num):
            # use relu between layers
            self.activations["Z"].append(np.add(np.dot(self.activations['A'][-1],\
                self.named_parameters['W_{}'.format(i)]), self.named_parameters['B_{}'.format(i)]))
            if i != self.layer_num - 1:
                self.activations["A"].append(relu(self.activations["Z"][-1]))
            else:
                self.activations["A"].append(softmax(self.activations["Z"][-1]))
        return self.activations["A"][-1]


class Adam(object):
    """
    Stochastic gradient descent optimizer with Adam
    Note: All default values are from the original Adam paper
    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params
    learning_rate_init : float, optional, default 0.1
        The initial learning rate used. It controls the step-size in updating
        the weights
    beta_1 : float, optional, default 0.9
        Exponential decay rate for estimates of first moment vector, should be
        in [0, 1)
    beta_2 : float, optional, default 0.999
        Exponential decay rate for estimates of second moment vector, should be
        in [0, 1)
    epsilon : float, optional, default 1e-8
        Value for numerical stability
    Attributes
    ----------
    learning_rate : float
        The current learning rate
    t : int
        Timestep
    ms : list, length = len(params)
        First moment vectors
    vs : list, length = len(params)
        Second moment vectors
    References
    ----------
    Kingma, Diederik, and Jimmy Ba.
    "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    """
    def __init__(self, params, learning_rate_init, beta_1, beta_2, epsilon):
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def _get_updates(self, grads):
        """Get the values used to update params with given gradients
        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates

    def update_params(self, params, grads):
        """Update parameters with given gradients
        Parameters
        ----------
        params: list
        grads : list, length = len(params)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        """
        updates = self._get_updates(grads)
        for param, update in zip(params, updates):
            param += update

def back_propagate(model, X, Y, num_classes):
    """
    Back popogate loss to each layer (proof/logic below)
    :param model: Dictionary with initialized weeights and biases
    :param X: Training Data, of shape [bs, dim]
    :param Y: True lable values of trainig data, of shape [bs, ]
    :param num_classes: number of classes, scalar
    :return: grads

    Forward pass: A[l] = act_func(A[l-1] @ W[l] + B[l])
    act_func: relu or softmax, l as the layer index
    Compute Derivatives: dLoss/dW[l], dLoss/dB[l]

    chain of thoughts:
    as, Z[l] = A[l-1] @ W[l] + B[l]
    so, dZ[l]/dW[l] = A[l-1]

    as, A[l] = act_func(Z[l])
    so, dA[l]/dZ[l] = act_func'(Z[l])

    Y_dist: one hot encode of Y, of shape [bs, num_classes]
    as, Loss = cross_entropy(A[l], Y)
    so, dLoss/dA[l] = 2 * (A[l] - Y_dist)

    so, dLoss/dW[l] = dLoss/dA[l] * dA[l]/dZ[l] * dZ[l]/dW[l]
                    = dLoss/dZ[l] * dZ[l]/dW[l]
    => dCost/dw[l] = A[l-1].transpose() @ (act_func'(Z[l]) * 2 * (A[l] - Y_dist))

    [32, 128] [128, 10] [32, 10]

    Similarly for bias:
    as, Z[l] = A[l-1] * W[l] + B[l]
    so, dZ[l]/dB[l] = 1
    so, dLoss/dW[l] = dLoss/dA[l] * dA[l]/dZ[l] * dZ[l]/dB[l]
                    = dLoss/dZ[l] * dZ[l]/dB[l]
    => dLoss/dB[l] = 1 * act_func'(Z[l]) * 2 * (A[l] - Y_dist)

    Since, this is for 1 example =>
    for n examples = 1/n * derivative
    """
    grads = []
    Y_hat = model(X)
    bs = Y_hat.shape[0]
    Y_dist = np.zeros(bs, num_classes, dtype=np.float)
    Y_dist[:, Y] = 1.
    loss = - np.log(Y_hat[:, Y]).sum()

    dZ = []
    dW = []
    dB = []
    for i in range(-1, -self.layer_num-1, -1):
        if len(dZ) == 0:
            dZ.append(1 / m * (model.activations["A"][-1] - Y_dist))
        else:
            dZ.append(1 / m * (np.dot(dZ[-1], np.transpose(model.parameters['W_{}'.format(self.layer_num+i)])) *\
                    (relu_derivative(model.activations["A"][i-1]))))
        dW.append(np.dot(np.transpose(model.activations["A"][i-1]), dZ[-1]))
        dB.append(dZ[-1].sum())
    grads.extend(dW)
    grads.extend(dB)
    return loss, grads

def fit_model(model, optimizer, X, Y, num_classes):
    loss, grads = back_propagate(model, X, Y, num_classes)
    optimizer.update_params(grads, model.parameters())



if __name__ == "__main__":
    layers = [128, 256, 256, 10]
    model = ToyModel(layers)
    dummy_input = np.random.rand(32, 128)
    out = model(dummy_input)
    print(out.shape)
