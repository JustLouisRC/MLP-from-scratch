import numpy as np
from mlp.base import Layer

class Linear(Layer):
    def __init__(self,input_dims, output_dims):
        limit = np.sqrt(6 / (input_dims + output_dims))
        self.weight = np.random.uniform(
            -limit,
            limit,
            (input_dims, output_dims)
        )
        self.bias = np.random.randn(1,output_dims) * 0.01

    def forward(self, input):
        self.input = input # (batch_size, input_dims)
        z = self.input @ self.weight + self.bias #(batch_size, output_dims)
        return z

    def backward(self, output_gradient, learning_rate):
        #output grad = (batch_size, output_dims)
        input_gradient = output_gradient @ self.weight.T
        # derivative of z = aw + b
        self.weight -= learning_rate * self.input.T @ output_gradient
        self.bias -= learning_rate * np.sum(output_gradient,axis = 0, keepdims = True)
        return input_gradient
