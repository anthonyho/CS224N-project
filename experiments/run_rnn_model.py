#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('../code'))
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import preprocess
import utils
import evaluate
import rnn_model


# Define global variables
embed_size = 50
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
class_weights = [1.0, 9.58871448, 1.810155, 31.99581504, 1.94160208, 10.88540896]
max_comment_size = 100

train_data_file = '../data/train.csv'
train_tokens_file = '../data/train_comments.p'

test_data_file = '../data/test.csv'
test_tokens_file = '../data/test_comments.p'

out_dir = 'out/'


# Define configs
debug = False

config = {'exp_name': 'rnn_full_10-1',
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


def load_and_process(train_data_file, test_data_file=None,
                     train_tokens_file=None, test_tokens_file=None,
                     debug=False):

    # Get glove/w2v data
    emb_data = preprocess.get_glove(embed_size)

    # Load and (optionally) subset train data
    train_data = preprocess.load_data(train_data_file, debug=debug)

    # Load test data
    if test_data_file:
        test_data = preprocess.load_data(test_data_file, debug=debug)
        id_test = test_data['id']
    else:
        id_test = None

    # Tokenize train comments or load pre-tokenized train comments
    if debug or (train_tokens_file is None):
        inputs, masks = preprocess.tokenize_df(train_data, target_length=max_comment_size)
    else:
        inputs = preprocess.load_tokenized_comments(train_tokens_file)  # <- to be fixed
    # Tokenize test comments or load pre-tokenized test comments
    if test_data_file:
        if test_tokens_file is None:
            inputs_test, masks_test = preprocess.tokenize_df(test_data, target_length=max_comment_size)
        else:
            inputs_test = preprocess.load_tokenized_comments(test_tokens_file)  # <- to be fixed
    else:
        inputs_test = None

    # Load train labels
    labels = preprocess.filter_labels(train_data, label_names)

    # Split to train and dev sets
    (inputs_train, labels_train, masks_train,
     inputs_dev, labels_dev, masks_dev) = preprocess.split_train_dev(inputs, labels, masks,
                                                                     fraction_dev=0.3)

    return (inputs_train, labels_train, masks_train,
            inputs_dev, labels_dev, masks_dev,
            id_test, inputs_test, masks_test), emb_data


def run(config, data, emb_data, debug=False):

    # Initialize
    print "Initializing..."
    (inputs_train, labels_train, masks_train,
     inputs_dev, labels_dev, masks_dev,
     id_test, inputs_test, masks_test) = data

    save_dir = os.path.join(out_dir, config['exp_name'])
    if debug:
        save_dir += '_debug'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_prefix = os.path.join(save_dir, config['exp_name'])

    # Fit
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        print "Building model..."
        obj = rnn_model.RNNModel(config, emb_data=emb_data)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        print "Training model..."
        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            (list_loss_train, list_loss_dev,
             list_score_train, list_score_dev,
             y_prob_train, y_prob_dev) = obj.train(sess,
                                                   inputs_train, masks_train, labels_train,
                                                   inputs_dev, masks_dev, labels_dev,
                                                   saver=saver, save_prefix=save_prefix)

    # Predict
    if inputs_test:
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            print "Rebuilding model..."
            obj = rnn_model.RNNModel(config, emb_data=emb_data)
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()

            print "Restoring model..."
            with tf.Session(graph=graph) as sess:
                sess.run(init_op)
                saver.restore(sess, save_prefix+'.weights')
                print "Predicting labels for test set..."
                y_prob_test = obj.predict(sess, inputs_test, masks_test)
        # Save y_prob_test to csv
        print "Saving test prediction..."
        y_prob_test_df = utils.y_prob_to_df(y_prob_test, id_test, label_names)
        y_prob_test_df.to_csv(save_prefix+'_test.csv', index=False)

    # Pack y_dict
    print "Evaluating..."
    y_dict = {'train': (labels_train, y_prob_train),
              'dev': (labels_dev, y_prob_dev)}

    # Evaluate, plot and save
    print "Final train loss = {:.4f}".format(list_loss_train[-1])
    print "Final dev loss = {:.4f}".format(list_loss_dev[-1])
    with open(save_prefix+'.txt', 'w') as f:
        yaml.dump(config, f)
        f.write('\n')
        f.write("Final train loss = {:.4f}\n".format(list_loss_train[-1]))
        f.write("Final dev loss = {:.4f}\n".format(list_loss_dev[-1]))
    evaluate.plot_loss(list_loss_train, list_loss_dev, save_prefix=save_prefix)
    evaluate.plot_score(list_score_train, list_score_dev, save_prefix=save_prefix)
    results_roc = evaluate.evaluate_full(y_dict, metric='roc', names=label_names,
                                         print_msg=True, save_msg=True, plot=True,
                                         save_prefix=save_prefix)
    results_prc = evaluate.evaluate_full(y_dict, metric='prc', names=label_names,
                                         print_msg=True, save_msg=True, plot=True,
                                         save_prefix=save_prefix)


if __name__ == '__main__':
    data, emb_data = load_and_process(train_data_file, test_data_file, debug=debug)
    for config in list_configs:
        run(config, data, emb_data, debug=debug)
