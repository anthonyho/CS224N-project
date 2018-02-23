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


def minibatch(batch_size, inputs_data, labels_data=None, shuffle=True):
    n_data = len(inputs_data)
    ind = np.arange(n_data)
    if shuffle:
        np.random.shuffle(ind)
    for i in np.arange(0, n_data, batch_size):
        inputs_batch =  _get_items(inputs_data, ind[i:i+batch_size])
        if labels_data is None:
            yield inputs_batch
        else:
            labels_batch = _get_items(labels_data, ind[i:i+batch_size])
            yield (inputs_batch, labels_batch)


def _get_items(data, ind):
    if isinstance(data, list):
        return [data[i] for i in ind]
    else:
        return data[ind]
