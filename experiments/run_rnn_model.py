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

config1 = {'exp_name': 'final_test_1',
          'n_epochs': 20,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 256,
          'lr': .0005,
          'batch_size': 256,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 2,
          'bidirectional': True,
          'averaging': False, #overwritten by attention
          'attention': True,
          'attention_size': 10,
          'sparsemax': False,
	  'clip_gradients': False,
          'max_grad_norm': 5          
}

config2 = {'exp_name': 'final_test_2',
          'n_epochs': 25,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 128,
          'lr': .0005,
          'batch_size': 512,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 2,
          'bidirectional': True,
          'averaging': False, #overwritten by attention
          'attention': True,
          'attention_size': 10,
          'sparsemax': False,
          'clip_gradients': False,
          'max_grad_norm': 5
}

config3 = {'exp_name': 'final_test_3',
          'n_epochs': 25,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 64,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 2,
          'bidirectional': True,
          'averaging': False, #overwritten by attention
          'attention': True,
          'attention_size': 10,
          'sparsemax': False,
          'clip_gradients': False,
          'max_grad_norm': 5
}

config4 = {'exp_name': 'final_test_4',
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
          'averaging': False, #overwritten by attention
          'attention': True,
          'attention_size': 10,
          'sparsemax': False,
          'clip_gradients': True,
          'max_grad_norm': 5
}

config5 = {'exp_name': 'final_test_5',
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
          'averaging': False, #overwritten by attention
          'attention': True,
          'attention_size': 10,
          'sparsemax': True,
          'clip_gradients': True,
          'max_grad_norm': 5
}

config6 = {'exp_name': 'final_test_6',
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
          'averaging': False, #overwritten by attention
          'attention': True,
          'attention_size': 10,
          'sparsemax': False,
          'clip_gradients': True,
          'max_grad_norm': 5
}

config7 = {'exp_name': 'final_test_7',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 50,
          'lr': .0005,
          'batch_size': 1024,
          'cell_type': 'RNN',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 2,
          'bidirectional': True,
          'averaging': False, #overwritten by attention
          'attention': True,
          'attention_size': 10,
          'sparsemax': False,
          'clip_gradients': True,
          'max_grad_norm': 5
}

config8 = {'exp_name': 'final_test_8',
          'n_epochs': 50,
          'embed_size': embed_size,
          'n_labels': len(label_names),
          'class_weights': class_weights,
          'max_comment_size': max_comment_size,
          'state_size': 128,
          'lr': .0005,
          'batch_size': 512,
          'cell_type': 'LSTM',
          'cell_kwargs': {},
          'dropout': True,
          'keep_prob': 0.5,
          'n_layers': 2,
          'bidirectional': True,
          'averaging': False, #overwritten by attention
          'attention': True,
          'attention_size': 10,
          'sparsemax': True,
          'clip_gradients': True,
          'max_grad_norm': 5
}

list_configs = [config1,config2,config3,config4,config5,config6,config7]


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
