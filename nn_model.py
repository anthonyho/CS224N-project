#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import preprocessing
import utils
from model import Model
import evaluate
import vocab


example_config = {'n_epochs': 500,  # number of iterations
                  'n_features': 50,  # dimension of the inputs
                  'n_labels': 3,  # number of labels to predict
                  'n_layers': 2,  # number of hidden layers
                  'hidden_sizes': [20, 20],  # size of hidden layers; int or list of int
                  'lr': .0005,  # learning rate
                  'batch_size': 1000,  # number of training examples in each minibatch
                  'activation': tf.nn.relu,
                  'optimizer': tf.train.AdamOptimizer,
                  'initializer': tf.contrib.layers.xavier_initializer(uniform=False)
                  }


class FeedForwardNeuralNetwork(Model):

    def _add_placeholders(self):
        input_shape = (None, self.config['n_features'])
        labels_shape = (None, self.config['n_labels'])
        self.input_placeholder = tf.placeholder(tf.float32,
                                                shape=input_shape)
        self.labels_placeholder = tf.placeholder(tf.float32,
                                                 shape=labels_shape)

    def _create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {self.input_placeholder: inputs_batch}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def _add_prediction_op(self):
        try:
            sizes = list(self.config['hidden_sizes'])
        except TypeError:
            sizes = [self.config['hidden_sizes']] * self.config['n_layers']
        assert len(sizes) == self.config['n_layers']
        sizes = [self.config['n_features']] + sizes + [self.config['n_labels']]

        activation = self.config['activation']
        initializer = self.config['initializer']

        h = self.input_placeholder

        for i in range(len(sizes) - 1):
            with tf.variable_scope('layer'+str(i)):
                W_shape = (sizes[i], sizes[i + 1])
                b_shape = (1, sizes[i + 1])
                W = tf.get_variable('W', shape=W_shape,
                                    initializer=initializer)
                b = tf.get_variable('b', shape=b_shape,
                                    initializer=initializer)
                if i == len(sizes) - 2:
                    z = tf.matmul(h, W) + b
                else:
                    z = tf.matmul(h, W) + b
                    h = activation(z)
        pred = z
        return pred

    def _add_loss_op(self, pred):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_placeholder,
                                                       logits=pred)
        loss = tf.reduce_mean(loss)
        return loss

    def _add_training_op(self, loss):
        opt = self.config['optimizer'](learning_rate=self.config['lr'])
        train_op = opt.minimize(loss)
        return train_op

    def _train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self._create_feed_dict(inputs_batch, labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def _run_epoch(self, sess, inputs, labels, shuffle):
        minibatches = utils.minibatch(self.config['batch_size'],
                                      inputs, labels, shuffle)
        loss = 0
        for i, (inputs_batch, labels_batch) in enumerate(minibatches):
            loss += self._train_on_batch(sess, inputs_batch, labels_batch)
        loss /= (i + 1)
        return loss

    def _transform_inputs(self, tokens):
        inds = preprocessing.tokens_to_ids(tokens, self.word2id)
        return preprocessing.average_sentence_vectors(inds, self.emb_matrix)

    def train(self, sess, tokens, labels, shuffle=True):
        inputs = self._transform_inputs(tokens)
        list_loss = []
        for epoch in range(self.config['n_epochs']):
            list_loss.append(self._run_epoch(sess, inputs, labels, shuffle))
        return list_loss

    def predict(self, sess, tokens):
        inputs = self._transform_inputs(tokens)
        feed = self._create_feed_dict(inputs)
        pred = sess.run(self.pred, feed_dict=feed)
        y_score = utils.sigmoid(pred)
        return y_score


def main():
    out_dir = 'out/'

    glove_file = 'data/glove/glove.6B.50d.txt'
    glove_dim = 50
    emb_data = vocab.get_glove(glove_file, glove_dim)

    data = preprocessing.load_data('data/train.csv')
    subset_data = data.head(10000)
    inputs = preprocessing.tokenize_df(subset_data)
    col = ['toxic', 'obscene', 'insult']
    labels = preprocessing.filter_labels(subset_data, col)
    inputs_train, labels_train, inputs_dev, labels_dev = preprocessing.split_train_dev(inputs, labels)

    tf.reset_default_graph()
    with tf.Graph().as_default() as graph: 
        obj = FeedForwardNeuralNetwork(example_config, emb_data=emb_data)
        init_op = tf.global_variables_initializer()
    graph.finalize()

    with tf.Session(graph=graph) as sess:
        sess.run(init_op)
        list_loss = obj.train(sess, inputs_train, labels_train)
        y_score_train = obj.predict(sess, inputs_train)
        y_score_dev = obj.predict(sess, inputs_dev)

    results_train = evaluate.evaluate(labels_train, y_score_train, names=col)
    results_dev = evaluate.evaluate(labels_dev, y_score_dev, names=col)
    
    print "Results on train set:\n"
    print results_train
    print "Results on dev set:\n"
    print results_dev

    np.save(out_dir+'y_true_train.npy', labels_train)
    np.save(out_dir+'y_score_train.npy', y_score_train)
    np.save(out_dir+'y_true_dev.npy', labels_dev)
    np.save(out_dir+'y_score_dev.npy', y_score_dev)


if __name__ == '__main__':
    main()
