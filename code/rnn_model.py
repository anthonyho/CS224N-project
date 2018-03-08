import numpy as np
import tensorflow as tf
from preprocess import get_glove, tokens_to_ids
from model import Model
import utils


example_config = {'exp_name': 'rnn_full_1',
                  'n_epochs': 50,  # number of iterations
                  'embed_size': 300,  # dimension of the inputs
                  'n_labels': 6,  # number of labels to predict
                  'max_comment_size'  : 250,
                  'state_size': 50,  # size of hidden layers; int
                  'lr': .001,  # learning rate
                  'batch_size': 1024,  # number of training examples in each minibatch
                  'cell_type': 'LSTM',
                  'cell_kwargs': {},
                  'dropout': True,
                  'dropout_rate': 0.5,
                  'n_layers': 2,
                  'bidirectional': True,
                  'averaging': True
                  }


class RNNModel(Model):

    def _add_placeholders(self):
        input_shape = (None, self.config['max_comment_size'])
        labels_shape = (None, self.config['n_labels'])
        mask_shape = (None, self.config['max_comment_size'])
        self.input_placeholder = tf.placeholder(tf.int32, shape=input_shape)
        self.labels_placeholder = tf.placeholder(tf.float32, shape=labels_shape)
        self.mask_placeholder = tf.placeholder(tf.bool, shape=mask_shape)

    def _create_feed_dict(self, inputs_batch, masks_batch, labels_batch=None):
        feed_dict = {self.input_placeholder: inputs_batch,
                     self.mask_placeholder: masks_batch}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def _add_prediction_op(self):
        # Transform ids to embeddings
        embed_matrix = tf.Variable(self.emb_matrix)
        embedded = tf.nn.embedding_lookup(embed_matrix, self.input_placeholder)
        x = tf.reshape(embedded, (-1, self.config['max_comment_size'], self.config['embed_size']))
        # Declare variable
        U = tf.get_variable('U', shape=(self.config['state_size'], self.config['n_labels']),
                            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', shape=(self.config['n_labels']),
                             initializer=tf.constant_initializer())
        # Create cells for each layer
        list_cells = []
        for i in range(self.config['n_layers']):
            if self.config['cell_type'] == 'RNN':
                list_cells.append(tf.contrib.rnn.BasicRNNCell(self.config['state_size'],
                                                              **self.config['cell_kwargs']))
            elif self.config['cell_type'] == 'GRU':
                list_cells.append(tf.contrib.rnn.GRUCell(self.config['state_size'],
                                                         **self.config['cell_kwargs']))
            elif self.config['cell_type'] == 'LSTM':
                list_cells.append(tf.contrib.rnn.LSTMCell(self.config['state_size'],
                                                          state_is_tuple=True,
                                                          **self.config['cell_kwargs']))
            else:
                raise NotImplementedError()
        # Unroll timesteps
        if self.config['bidirectional']:
            # Shape of outputs = (batch_size, max_length, state_size * 2)
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(list_cells, list_cells,
                                                                           x, dtype=tf.float32)
            final_output_fw = self._agg_outputs(outputs[:, :, :self.config['state_size']],
                                                self.config['averaging'])
            final_output_bw = self._agg_outputs(outputs[:, :, self.config['state_size']:],
                                                self.config['averaging'])
            final_output = (final_output_fw + final_output_bw) / 2
        else:
            # Create layers using MultiRNNCell()
            if self.config['cell_type'] == 'LSTM':
                multi_cells = tf.contrib.rnn.MultiRNNCell(list_cells, state_is_tuple=True)
            else:
                multi_cells = tf.contrib.rnn.MultiRNNCell(list_cells)
            # Shape of outputs = (batch_size, max_length, state_size)
            outputs, state = tf.nn.dynamic_rnn(multi_cells, x, dtype=tf.float32)
            final_output = self._agg_outputs(outputs, self.config['averaging'])
        # Add droput layer
        if self.config['dropout']:
            final_output_dropout = tf.nn.dropout(final_output, self.config['dropout_rate'])
        else:
            final_output_dropout = final_output
        # Final layer
        pred = tf.matmul(final_output_dropout, U) + b2
        return pred

    def _agg_outputs(self, outputs, averaging=True):
        if averaging:
            mask = tf.cast(self.mask_placeholder, tf.float32)
            n_words = tf.reduce_sum(mask, axis=1, keepdims=True)
            mask_stack = tf.stack([mask] * outputs.shape[-1], axis=-1)
            sum_outputs = tf.reduce_sum(outputs * mask_stack, axis=1)
            final_output = sum_outputs / n_words
        else:
            mask = tf.cast(self.mask_placeholder, tf.int32)
            n_words = tf.reduce_sum(mask, axis=1)
            idx = tf.range(tf.shape(self.input_placeholder)[0]) * tf.shape(outputs)[1] + (n_words - 1)
            final_output = tf.gather(tf.reshape(outputs, (-1, self.config['state_size'])), idx)
        return final_output

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

    def _evaluate_on_batch(self, sess, inputs_batch, masks_batch, labels_batch):
        feed = self._create_feed_dict(inputs_batch, masks_batch, labels_batch)
        loss = sess.run(self.loss, feed_dict=feed)
        return loss

    def _predict_on_batch(self, sess, inputs_batch, masks_batch):
        feed = self._create_feed_dict(inputs_batch, masks_batch)
        pred_batch = sess.run(self.pred, feed_dict=feed)
        return utils.sigmoid(pred_batch)

    def _run_epoch_train(self, sess, inputs, masks, labels, shuffle):
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

    def _run_epoch_dev(self, sess, inputs, masks, labels, shuffle):
        n_minibatches = 1 + int(len(inputs) / self.config['batch_size'])
        minibatches = utils.minibatch(self.config['batch_size'],
                                      inputs, labels=labels, masks=masks, shuffle=shuffle)
        loss = 0
        for i, (inputs_batch, masks_batch, labels_batch) in enumerate(minibatches):
            loss += self._evaluate_on_batch(sess, inputs_batch, masks_batch, labels_batch)
        loss /= (i + 1)
        print 'dev loss = {:.4f}'.format(loss)
        return loss

    def train(self, sess,
              tokens_train, masks_train, labels_train,
              tokens_dev=None, masks_dev=None, labels_dev=None,
              shuffle=True):
        inputs_train = np.array(tokens_to_ids(tokens_train, self.word2id))
        list_train_loss = []
        if tokens_dev is not None:
            inputs_dev = np.array(tokens_to_ids(tokens_dev, self.word2id))
            list_dev_loss = []
        for epoch in range(self.config['n_epochs']):
            print "Epoch = {}/{}:".format(str(epoch + 1), str(self.config['n_epochs']))
            list_train_loss.append(self._run_epoch_train(sess,
                                                         inputs_train, masks_train, labels_train,
                                                         shuffle))
            if tokens_dev is not None:
                list_dev_loss.append(self._run_epoch_dev(sess,
                                                         inputs_dev, masks_dev, labels_dev,
                                                         shuffle))
        if tokens_dev is not None:
            return list_train_loss
        else:
            return list_train_loss, list_dev_loss

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
