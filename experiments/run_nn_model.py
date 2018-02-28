#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.abspath('../code'))
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import preprocess
import evaluate
import nn_model


config = {'exp_name': 'ff_l2_h30_f300',
          'label_names': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
          'n_epochs': 500,  # number of iterations
          'n_features': 300,  # dimension of the inputs
          'n_labels': 6,  # number of labels to predict
          'n_layers': 2,  # number of hidden layers
          'hidden_sizes': [30, 30],  # size of hidden layers; int or list of int
          'lr': .0005,  # learning rate
          'batch_size': 2000,  # number of training examples in each minibatch
          'activation': tf.nn.relu,
          'optimizer': tf.train.AdamOptimizer,
          'initializer': tf.contrib.layers.xavier_initializer(uniform=False)
          }


train_data_file = '../data/train.csv'
train_tokens_file = '../data/train_comments.p'

test_data_file = '../data/test.csv'
test_tokens_file = '../data/test_comments.p'

out_dir = 'out/'


def load_and_process(config, train_data_file, test_data_file=None,
                     train_tokens_file=None, test_tokens_file=None,
                     debug=False):

    # Get glove/w2v data
    emb_data = preprocess.get_glove(config['n_features'])

    # Load and (optionally) subset train data
    train_data = preprocess.load_data(train_data_file)
    if debug and isinstance(debug, bool):
        train_data = train_data.head(6000)
    elif debug and isinstance(debug, int):
        train_data = train_data.head(debug)

    # Load test data
    if test_data_file:
        test_data = preprocess.load_data(test_data_file)
        id_test = test_data['id']
    else:
        id_test = None

    # Tokenize train comments or load pre-tokenized train comments
    if debug or (train_tokens_file is None):
        inputs = preprocess.tokenize_df(train_data)
    else:
        inputs = preprocess.load_tokenized_comments(train_tokens_file)
    # Tokenize test comments or load pre-tokenized test comments
    if test_data_file:
        if test_tokens_file is None:
            inputs_test = preprocess.tokenize_df(test_data)
        else:
            inputs_test = preprocess.load_tokenized_comments(test_tokens_file)
    else:
        inputs_test = None

    # Load train labels
    labels = preprocess.filter_labels(train_data, config['label_names'])

    # Split to train and dev sets
    inputs_train, labels_train, inputs_dev, labels_dev = preprocess.split_train_dev(inputs, labels,
                                                                                    fraction_dev=0.3)

    return (inputs_train, labels_train, inputs_dev, labels_dev, inputs_test, id_test), emb_data


def run(config, data, emb_data, debug=False):

    # Unpack data
    (inputs_train, labels_train, inputs_dev, labels_dev, inputs_test, id_test) = data

    # Initialize graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        obj = nn_model.FeedForwardNeuralNetwork(config, emb_data=emb_data)
        init_op = tf.global_variables_initializer()
    graph.finalize()

    # Fit
    with tf.Session(graph=graph) as sess:
        sess.run(init_op)
        list_loss = obj.train(sess, inputs_train, labels_train)
        y_score_train = obj.predict(sess, inputs_train)
        y_score_dev = obj.predict(sess, inputs_dev)
        if inputs_test:
            y_score_test = obj.predict(sess, inputs_test)

    # Pack y_dict
    y_dict = {'train': (labels_train, y_score_train),
              'dev': (labels_dev, y_score_dev)}

    # Evaluate, plot and save
    save_prefix = os.path.join(out_dir, config['exp_name'])
    if debug:
        save_prefix += '_debug'
    print 'Final loss = {:.4f}'.format(list_loss[-1])
    with open(save_prefix+'.txt', 'w') as f:
        f.write('Final loss = {:.4f}\n'.format(list_loss[-1]))
    evaluate.plot_loss(list_loss, save_prefix=save_prefix)
    results_roc = evaluate.evaluate_full(y_dict, metric='roc', names=config['label_names'],
                                         print_msg=True, save_msg=True, plot=True,
                                         save_prefix=save_prefix)
    results_prc = evaluate.evaluate_full(y_dict, metric='prc', names=config['label_names'],
                                         print_msg=True, save_msg=True, plot=True,
                                         save_prefix=save_prefix)

    # Save y_score_test to csv
    if inputs_test:
        y_score_test_df = pd.DataFrame(y_score_test, columns=config['label_names'])
        y_score_test_df = pd.concat([id_test, y_score_test_df], axis=1)
        y_score_test_df.fillna(0.5).to_csv(save_prefix+'_test.csv', index=False) # quick hack

if __name__ == '__main__':
    debug = False
    data, emb_data = load_and_process(config, train_data_file, test_data_file,
                                      train_tokens_file, test_tokens_file,
                                      debug=debug)
    run(config, data, emb_data, debug=debug)
