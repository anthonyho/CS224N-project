#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('../code'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rnn_model


# Define global variables
embed_size = 50
max_comment_size = 100
fraction_dev = 0.3
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
class_weights = [1.0, 9.58871448, 1.810155, 31.99581504, 1.94160208, 10.88540896]

train_data_file = '../data/train.csv'
train_tokens_file = '../data/train_comments.p'

test_data_file = '../data/test.csv'
test_tokens_file = '../data/test_comments.p'

out_dir = 'out/'


# Define configs
debug = 5000

config = {'exp_name': 'rnn_full_1',
          'n_epochs': 8,
          'embed_size': embed_size,
          'n_labels': 6,
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .001,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 1,
          'bidirectional': False,
          'averaging': True
          }

list_configs = [config]


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
