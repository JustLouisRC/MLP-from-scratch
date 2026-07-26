import numpy as np
from mlp.base import Activation


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(output):
            return 1 - output**2

        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    def __init__(self):
        def relu(x): return np.maximum(0,x)
        def relu_prime(output):return (output>0).astype(float)

        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(output):
            return output * (1 - output)

        super().__init__(sigmoid, sigmoid_prime)