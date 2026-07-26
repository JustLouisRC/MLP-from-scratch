class Layer:
    def __init__(self):
        pass
    def forward(self,input_data):
        pass
    def backward(self,gradient, learning_rate):
        pass

class Activation(Layer):
    def __init__(self, func, func_prime):
        self.activation = func
        self.activation_prime = func_prime
    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output
    def backward(self, gradient, learning_rate=None):
        return gradient * self.activation_prime(self.output)

class Loss:
    def __init__(self, loss_func, loss_func_prime):
        self.loss_func = loss_func
        self.loss_func_prime = loss_func_prime

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        pred, loss = self.loss_func(y_pred, y_true)
        self.pred = pred
        return pred, loss

    def backward(self):
        return self.loss_func_prime(self.y_pred, self.y_true)