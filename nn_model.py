#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import preprocessing
from model import Model


config = {'n_epochs': 10,
          'n_features': 50,
          'n_classes': 2,
          'hidden_size': 20,
          'lr': .0005
          }


class FeedForwardNeuralNetwork(Model):

    def add_placeholders(self):
        input_shape = (None, self.config['n_features'])
        labels_shape = (None, self.config['n_classes'])
        self.input_placeholder = tf.placeholder(tf.float32,
                                                shape=input_shape)
        self.labels_placeholder = tf.placeholder(tf.float32,
                                                 shape=labels_shape)

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {self.input_placeholder: inputs_batch}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        W1 = tf.get_variable('W1',
                             shape=[self.config['n_features'], self.config['hidden_size']],
                             initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b1 = tf.Variable(tf.zeros((1, self.config['hidden_size'])),
                         dtype=tf.float32)
        h = tf.nn.relu(tf.matmul(self.input_placeholder, W1) + b1)
        W2 = tf.get_variable('W2',
                             shape=[self.config['hidden_size'], self.config['n_classes']],
                             initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b2 = tf.Variable(initial_value=tf.zeros((1, self.config['n_classes'])),
                         dtype=tf.float32)
        pred = tf.matmul(h, W2) + b2
        return pred

    def add_loss_op(self, pred):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_placeholder,
                                                       logits=pred)
        loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self, loss):
        opt = tf.train.AdamOptimizer(learning_rate=self.config['lr'])
        train_op = opt.minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, sentence_avgs, array_labels):
        loss = self.train_on_batch(sess, sentence_avgs, array_labels)
        return loss

    def train(self, sess, list_list_tokens, array_labels):
        list_list_inds = preprocessing.tokens_to_ids(list_list_tokens, self.word2id)
        sentence_avgs = preprocessing.average_sentence_vectors(list_list_inds, self.emb_matrix)
        list_loss = []
        for epoch in range(self.config['n_epochs']):
            list_loss.append(self.run_epoch(sess, sentence_avgs, array_labels))
        return list_loss

    def predict(self, sess, list_list_tokens):
        list_list_inds = preprocessing.tokens_to_ids(list_list_tokens, self.word2id)
        sentence_avgs = preprocessing.average_sentence_vectors(list_list_inds, self.emb_matrix) 
        feed = self.create_feed_dict(sentence_avgs)
        classification = sess.run(self.pred, feed_dict=feed)
        return classification



def main():
    out_dir = 'out/'
    
    train_data = preprocessing.load_data('data/train.csv')
    subset_train_data = train_data.head(5000)
    list_list_tokens = preprocessing.tokenize_df(subset_train_data)
    array_labels = preprocessing.filter_labels(subset_train_data, ['toxic', 'severe_toxic'])
    
    with tf.Graph().as_default() as graph: 
        model = LogisticRegressionModel(config['n_features'], config)
        init_op = tf.global_variables_initializer()
    graph.finalize()
    
    with tf.Session(graph=graph) as sess:
        sess.run(init_op)
        list_loss = model.train(sess, list_list_tokens, array_labels)
        y_score = preprocessing.sigmoid(model.predict(sess, list_list_tokens))
    
    np.save(out_dir+'y_true.npy', array_labels)
    np.save(out_dir+'y_score.npy', y_score)


if __name__ == '__main__':
    main()
