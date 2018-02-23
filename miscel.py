import numpy as np


def softmax(x):
    if len(x.shape) > 1:
        x = np.exp(x - np.max(x, axis=1, keepdims=True))
        x = x / np.sum(x, axis=1, keepdims=True)
    else:
        x = np.exp(x - np.max(x))
        x = x / np.sum(x)
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
