#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.abspath('../code'))
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import preprocess
import evaluate
import nn_model


config = {'exp_name': 'ff_l3_h30_f300',
          'label_names': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
          'n_epochs': 500,  # number of iterations
          'n_features': 300,  # dimension of the inputs
          'n_labels': 6,  # number of labels to predict
          'n_layers': 3,  # number of hidden layers
          'hidden_sizes': [30, 30, 30],  # size of hidden layers; int or list of int
          'lr': .0005,  # learning rate
          'batch_size': 2000,  # number of training examples in each minibatch
          'activation': tf.nn.relu,
          'optimizer': tf.train.AdamOptimizer,
          'initializer': tf.contrib.layers.xavier_initializer(uniform=False)
          }

data_file = '../data/train.csv'
tokenized_comments_file = '../data/train_comments.p'
out_dir = 'out/'


def load_and_process(config, data_file, tokenized_comments_file=None, debug=False):

    # Get glove/w2v data
    emb_data = preprocess.get_glove(config['n_features'])

    # Load and (optionally) subset data
    data = preprocess.load_data(data_file)
    if debug and isinstance(debug, bool):
        data = data.head(6000)
    elif debug and isinstance(debug, int):
        data = data.head(debug)

    # Tokenize comments or load pre-tokenized comments
    if debug or (tokenized_comments_file is None):
        inputs = preprocess.tokenize_df(data)
    else:
        inputs = preprocess.load_tokenized_comments(tokenized_comments_file)

    # Load labels
    labels = preprocess.filter_labels(data, config['label_names'])

    # Split to train and dev sets
    inputs_train, labels_train, inputs_dev, labels_dev = preprocess.split_train_dev(inputs, labels,
                                                                                    fraction_dev=0.3)

    return (inputs_train, labels_train, inputs_dev, labels_dev), emb_data


def run(config, data, emb_data, debug=False):

    # Unpack data
    (inputs_train, labels_train, inputs_dev, labels_dev) = data

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

    # Pack y_dict
    y_dict = {'train': (labels_train, y_score_train),
              'dev': (labels_dev, y_score_dev)}

    # Evaluate, plot and save
    file_prefix = os.path.join(out_dir, config['exp_name'] + '_')
    if debug:
        file_prefix += 'debug_'
    evaluate.plot_loss(list_loss, fig_path=file_prefix+'loss')
    results_roc = evaluate.evaluate_full(y_dict, names=config['label_names'],
                                         metric='roc', fig_path=file_prefix+'roc')
    results_prc = evaluate.evaluate_full(y_dict, names=config['label_names'],
                                         metric='prc', fig_path=file_prefix+'prc')


if __name__ == '__main__':
    debug = False
    data, emb_data = load_and_process(config, data_file, tokenized_comments_file, debug=debug)
    run(config, data, emb_data, debug=debug)
