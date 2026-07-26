import numpy as np
class MLP:
    def __init__(self, network, loss_function, learning_rate):
        self.network = network
        self.loss_function = loss_function
        self.learning_rate = learning_rate
    def forward(self, input_data):
        act = input_data
        for layer in self.network:
            act = layer.forward(act)
        return act
    def backward(self):
        grad = self.loss_function.backward()
        for layer in reversed(self.network):
            grad = layer.backward(grad,self.learning_rate)

    def fit(self, X, Y, epochs = 500, batch_size = 20, print_every = 100):
        for e in range(epochs+1):
            epoch_loss = 0
            epoch_step = 0
            epoch_acc = 0
            for i in range(0,X.shape[0], batch_size):
                x_input = X[i : i+batch_size]
                y_input = Y[i : i+batch_size]
                # Forward
                pred = self.forward(x_input)
                # Loss
                pred, loss = self.loss_function.forward(pred,y_input)
                epoch_step+=1
                epoch_loss+=loss
                epoch_acc += np.mean(np.argmax(pred, axis=1) == y_input)
                # Backward
                self.backward()
            if(e%print_every == 0):
                print(f"epoch : {e} | loss : {epoch_loss / epoch_step} | accuracy : {epoch_acc / epoch_step}")
    def predict(self, X):
        pred = self.forward(X)
        return np.argmax(pred, axis=1)