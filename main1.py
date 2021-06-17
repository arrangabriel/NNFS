import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

x, y = spiral_data(samples=100, classes=3)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')
plt.show()

# fully-connected layer
class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # self.output = 0

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # return self.output


# ReLU activation class
class ActivationReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Softmax activation class
class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# common loss class
class Loss:

    # calculates losses given outpupt and ground truth values
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# cross-entropy loss
class LossCategoricalEntropy(Loss):

    # forward pass
    def forward(self, y_pred, y_true):
        # number of samples in a batch
        samples = len(y_pred)

        # clip data to prevent division by 0
        # clip both sides to not drag mean to any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]    # understand this line better
        # one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # losses
        negative_log_likelyhoods = -np.log(correct_confidences)
        return negative_log_likelyhoods


# Layer 1
dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

# Layer 2
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

# Loss
loss_function = LossCategoricalEntropy()

dense1.forward(x)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y)

print(activation2.output[:5])
print('Loss: ', loss)