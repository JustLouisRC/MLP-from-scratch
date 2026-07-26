from mlp.base import Loss
import numpy as np
class SoftMaxCategoricalEntropy(Loss):
    def __init__(self, temp=1.0, num_classes=2):
        self.temp = temp
        self.num_classes = num_classes
        def softmax_ce(y_pred, y_true):
            # SoftMax
            y_scaled = y_pred / self.temp
            y_scaled_max = np.max(y_scaled, axis=1, keepdims=True)
            e = np.exp(y_scaled - y_scaled_max)
            softmax = e / np.sum(e, axis=1, keepdims=True)

            self.softmax = softmax
            self.batch_size = softmax.shape[0]
            self.y_true_onehot = np.eye(self.num_classes)[y_true]
            # CE
            probs_clipped = np.clip(softmax, 1e-12, 1 - 1e-12)
            loss = -np.sum(self.y_true_onehot * np.log(probs_clipped)) / self.batch_size
            return softmax, loss

        def softmax_ce_prime(y_pred, y_true):
            return (self.softmax - self.y_true_onehot) / self.batch_size

        super().__init__(softmax_ce, softmax_ce_prime)