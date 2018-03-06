import tensorflow as tf
from preprocess import get_glove, tokens_to_ids
from model import Model
import utils
import numpy as np


config = {'exp_name': 'rnn_full_1',
          'label_names': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
          'n_epochs': 50,  # number of iterations
          'embed_size': 300,  # dimension of the inputs
          'n_features': 300,  # dimension of the inputs
          'n_labels': 6,  # number of labels to predict
          'max_comment_size'  : 250,
          'state_size': 50,  # size of hidden layers; int
          'lr': .001,  # learning rate
          'batch_size': 2048,  # number of training examples in each minibatch
          'cell': tf.contrib.rnn.LSTMCell,
          'cell_kwargs': {},
          'dropout': True,
          'dropout_kwargs': {'input_keep_prob': 0.8,
                             'output_keep_prob': 0.8,
                             'state_keep_prob': 0.8},
          'n_layers': 2,
          'bidirectional': False
          }


class RNNModel(Model):

    def _add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32,
                                                shape=(None, self.config['max_comment_size']))
        self.labels_placeholder = tf.placeholder(tf.float32,
                                                 shape=(None, self.config['n_labels']))
        self.mask_placeholder = tf.placeholder(tf.bool,
                                               shape=(None,self.config['max_comment_size']))

    def _create_feed_dict(self, inputs_batch, masks_batch, labels_batch=None):
        feed_dict = {self.input_placeholder: inputs_batch,
                     self.mask_placeholder: masks_batch}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def _add_prediction_op(self):
        embed_matrix = tf.Variable(self.emb_matrix.astype('float32'))
        embeddings2 = tf.nn.embedding_lookup(embed_matrix, self.input_placeholder)
        embeddings = tf.reshape(embeddings2, (-1, self.config['max_comment_size'], self.config['embed_size']))
        
        U = tf.get_variable("U",shape=(self.config['state_size'],self.config['n_labels']),initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2",shape=(self.config['n_labels']),initializer=tf.constant_initializer())
        # Create cell
        x = embeddings
        cell = self.config['cell'](self.config['state_size'], **self.config['cell_kwargs'])
        # Add droput
        if dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell, **self.config['dropout_kwargs'])
        # Create layers
        multi_cells = tf.contrib.rnn.MultiRNNCell([cell] * self.config['n_layers'])
        # Unroll
        if bidirectional:
            outputs, state = tf.nn.bidirectional_dynamic_rnn(multi_cells, multi_cells, x, dtype=tf.float32)
        else:
            outputs, state = tf.nn.dynamic_rnn(multi_cells, x, dtype=tf.float32)
        seq_lengths = tf.reduce_sum(tf.cast(self.mask_placeholder, tf.int32), axis=1)
        idx = tf.range(tf.shape(self.input_placeholder)[0])*tf.shape(outputs)[1] + (seq_lengths - 1)
        last_rnn_outputs = tf.gather(tf.reshape(outputs, [-1, self.config['state_size']]), idx)
        
        pred = tf.matmul(last_rnn_outputs,U) + b2
        return pred

    def _add_loss_op(self, pred):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,
                                                       labels=self.labels_placeholder)
        return tf.reduce_mean(loss)

    def _add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config['lr'])
        train_op = optimizer.minimize(loss)
        return train_op

    def _train_on_batch(self, sess, inputs_batch, masks_batch, labels_batch):
        feed = self._create_feed_dict(inputs_batch, masks_batch, labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def _predict_on_batch(self, sess, inputs_batch, masks_batch):
        feed = self._create_feed_dict(inputs_batch, masks_batch)
        pred_batch = sess.run(self.pred, feed_dict=feed)
        return utils.sigmoid(pred_batch)

    def _run_epoch(self, sess, inputs, masks, labels, shuffle):
        n_minibatches = 1 + int(len(inputs) / self.config['batch_size'])
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        minibatches = utils.minibatch(self.config['batch_size'],
                                      inputs, labels=labels, masks=masks, shuffle=shuffle)
        loss = 0
        for i, (inputs_batch, masks_batch, labels_batch) in enumerate(minibatches):
            loss += self._train_on_batch(sess, inputs_batch, masks_batch, labels_batch)
            prog.update(i+1,[('train_loss', loss)], force=i+1 == n_minibatches)
        loss /= (i + 1)
        return loss

    def train(self, sess, tokens, masks, labels, shuffle=True):
        inputs = np.array(tokens_to_ids(tokens, self.word2id))
        list_loss = []
        for epoch in range(self.config['n_epochs']):
            print "Epoch = {}/{}:".format(str(epoch + 1), str(self.config['n_epochs']))
            list_loss.append(self._run_epoch(sess, inputs, masks, labels, shuffle))
        return list_loss

    def predict(self, sess, tokens, masks):
        inputs = np.array(tokens_to_ids(tokens, self.word2id))
        minibatches = utils.minibatch(self.config['batch_size'],
                                      inputs, labels=None, masks=masks, shuffle=False)
        list_y_score = []
        for i, (inputs_batch, masks_batch) in enumerate(minibatches):
            list_y_score.append(self._predict_on_batch(sess, inputs_batch, masks_batch))
        y_score = np.vstack(list_y_score)
        return y_score

    def save_weights(self, file_path):
        raise NotImplementedError()

    def load_weights(self, file_path):
        raise NotImplementedError()

    def build(self):
        self._add_placeholders()
        self.pred = self._add_prediction_op()
        self.loss = self._add_loss_op(self.pred)
        self.train_op = self._add_training_op(self.loss)
