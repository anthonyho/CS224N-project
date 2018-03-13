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
max_comment_size = 250
fraction_dev = 0.3
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
class_weights = [1.0, 9.58871448, 1.810155, 31.99581504, 1.94160208, 10.88540896]

train_data_file = '../data/train.csv'
train_tokens_file = None#'../data/train_comments.p'

test_data_file = '../data/test.csv'
test_tokens_file = None#'../data/test_comments.p'

out_dir = 'out/'


# Define configs
debug = False

config1 = {'exp_name': 'new_test_1',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 1,
          'bidirectional': False,
          'averaging': True
          }

config2 = {'exp_name': 'new_test_2',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 100,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 1,
          'bidirectional': False,
          'averaging': True
          }

config3 = {'exp_name': 'new_test_3',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 2,
          'bidirectional': False,
          'averaging': True
          }

config4 = {'exp_name': 'new_test_4',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 1,
          'bidirectional': True,
          'averaging': True
          }

config5 = {'exp_name': 'new_test_5',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 2,
          'bidirectional': True,
          'averaging': True
          }

config6 = {'exp_name': 'new_test_6',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'GRU',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 1,
          'bidirectional': False,
          'averaging': True
          }

config7 = {'exp_name': 'new_test_7',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'GRU',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 1,
          'bidirectional': True,
          'averaging': True,
	  'attention': False,
	  'attention_size': 10
          }

config8 = {'exp_name': 'new_test_8',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'GRU',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 2,
          'bidirectional': True,
          'averaging': True,
	  'attention': False,
          'attention_size': 10
          }

config9 = {'exp_name': 'new_test_9',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 1,
          'bidirectional': False,
          'averaging': True,
	  'attention': True,
          'attention_size': 10
          }

config10 = {'exp_name': 'new_test_10',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 2,
          'bidirectional': False,
          'averaging': True,
	  'attention': True,
          'attention_size': 10
          }

config11 = {'exp_name': 'new_test_11',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 1,
          'bidirectional': True,
          'averaging': True,
	  'attention': True,
          'attention_size': 10
          }

config12 = {'exp_name': 'new_test_12',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 2,
          'bidirectional': True,
          'averaging': True,
          'attention': True,
          'attention_size': 10,
            'sparsemax': False
          }

list_configs = [config7,config8,config9,config10,config11,config12]


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
