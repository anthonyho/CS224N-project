import os
import sys
sys.path.insert(0, os.path.abspath('../code'))
import preprocess
import rnn_model


# Define global variables
embed_size = 300
max_comment_size = 250
fraction_dev = 0.3
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
class_weights = [1.0, 9.58871448, 1.810155, 31.99581504, 1.94160208, 10.88540896]

out_dir = 'out/'


# Define configs
config = {'exp_name': 'final_test_1_big',
          'n_epochs': 20,
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
          'averaging': False,
          'attention': True,
          'attention_size': 10,
          'sparsemax': False,
          'clip_gradients': False,
          'max_grad_norm': 5
          }


def load_model():
    save_prefix = os.path.join(out_dir, config['exp_name'], config['exp_name'])
    emb_data = preprocess.get_glove(embed_size)
    model = rnn_model.PredictWithRNNModel(config, emb_data, save_prefix)
    return model
