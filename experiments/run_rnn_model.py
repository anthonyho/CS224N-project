#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('../code'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import rnn_model


# Define global variables
embed_size = 300
max_comment_size = 500
fraction_dev = 0.3
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
class_weights = [1.0, 9.58871448, 1.810155, 31.99581504, 1.94160208, 10.88540896]

train_data_file = '../data/train.csv'
train_tokens_file = '../data/train_comments.p'

test_data_file = '../data/test.csv'
test_tokens_file = '../data/test_comments.p'

out_dir = 'out/'


# Define configs
debug = 4000

config1 = {'exp_name': 'rnn_test',
           'n_epochs': 50,
           'embed_size': embed_size,
           'n_labels': len(label_names),
           'class_weights': class_weights,
           'max_comment_size': max_comment_size,
           'state_size': 512,
           'lr': .0005,
           'batch_size': 128,
           'cell_type': 'LSTM',
           'cell_kwargs': {},
           'dropout': True,
           'keep_prob': 0.5,
           'n_layers': 1,
           'bidirectional': True,
           'averaging': False,  # overwritten by attention
           'attention': True,
           'attention_size': 10,
           'clip_gradients': True,
           'max_grad_norm': 5
           }

list_configs = [config1]


if __name__ == '__main__':
    data = rnn_model.load_and_process(train_data_file, test_data_file,
                                      train_tokens_file, test_tokens_file,
                                      embed_size=embed_size,
                                      max_comment_size=max_comment_size,
                                      label_names=label_names,
                                      fraction_dev=fraction_dev, debug=debug)
    for config in list_configs:
        rnn_model.run(config, *data,
                      label_names=label_names, out_dir=out_dir, debug=debug)
