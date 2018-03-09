import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import (BasicRNNCell, GRUCell, LSTMCell,
                                    MultiRNNCell,
                                    stack_bidirectional_dynamic_rnn)
from preprocess import get_glove, tokens_to_ids
from model import Model
import utils


example_config = {'exp_name': 'rnn_full_1',
                  'n_epochs': 50,
                  'embed_size': 300,
                  'n_labels': 6,
                  'class_weights': np.array([1.0, 9.58871448, 1.810155,
                                             31.99581504, 1.94160208,
                                             10.88540896]),
                  'max_comment_size': 250,
                  'state_size': 50,
                  'lr': .001,
                  'batch_size': 1024,
                  'cell_type': 'LSTM',
                  'cell_kwargs': {},
                  'dropout': True,
                  'keep_prob': 0.5,
                  'n_layers': 2,
                  'bidirectional': True,
                  'averaging': True
                  }


class RNNModel(Model):

    def _add_placeholders(self):
        input_shape = (None, self.config.max_comment_size)
        labels_shape = (None, self.config.n_labels)
        mask_shape = (None, self.config.max_comment_size)
        self.input_placeholder = tf.placeholder(tf.int32, input_shape)
        self.labels_placeholder = tf.placeholder(tf.float32, labels_shape)
        self.mask_placeholder = tf.placeholder(tf.bool, mask_shape)

    def _create_feed_dict(self, inputs_batch, masks_batch, labels_batch=None):
        feed_dict = {self.input_placeholder: inputs_batch,
                     self.mask_placeholder: masks_batch}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def _add_embeddings(self):
        embed_matrix = tf.Variable(self.emb_matrix)
        emb = tf.nn.embedding_lookup(embed_matrix, self.input_placeholder)
        emb_shape = (-1, self.config.max_comment_size, self.config.embed_size)
        return tf.reshape(emb, emb_shape)

    def _add_prediction_op(self):
        # Add embedding layer
        x = self._add_embeddings()
        # Create cells for each layer
        list_cells = []
        for i in range(self.config.n_layers):
            if self.config.cell_type == 'RNN':
                cell = BasicRNNCell(self.config.state_size,
                                    **self.config.cell_kwargs)
            elif self.config.cell_type == 'GRU':
                cell = GRUCell(self.config.state_size,
                               **self.config.cell_kwargs)
            elif self.config.cell_type == 'LSTM':
                cell = LSTMCell(self.config.state_size,
                                state_is_tuple=True, **self.config.cell_kwargs)
            else:
                raise NotImplementedError('Use cell_type = {RNN|GRU|LSTM}')
            list_cells.append(cell)
        # Unroll RNN time steps
        if self.config.bidirectional:
            # Shape of outputs = (batch_size, max_length, state_size * 2)
            outputs, _, _ = stack_bidirectional_dynamic_rnn(list_cells,
                                                            list_cells,
                                                            x,
                                                            dtype=tf.float32)
            h_fw = self._agg_outputs(outputs[:, :, :self.config.state_size],
                                     self.config.averaging)
            h_bw = self._agg_outputs(outputs[:, :, self.config.state_size:],
                                     self.config.averaging)
            h = (h_fw + h_bw) / 2
        else:
            # Create layers using MultiRNNCell()
            if self.config.cell_type == 'LSTM':
                multi_cells = MultiRNNCell(list_cells, state_is_tuple=True)
            else:
                multi_cells = MultiRNNCell(list_cells)
            # Shape of outputs = (batch_size, max_length, state_size)
            outputs, _ = tf.nn.dynamic_rnn(multi_cells, x, dtype=tf.float32)
            h = self._agg_outputs(outputs, self.config.averaging)
        # Add droput layer
        if self.config.dropout:
            h_dropout = tf.nn.dropout(h, self.config.keep_prob)
        else:
            h_dropout = h
        # Final layer
        U_shape = (self.config.state_size, self.config.n_labels)
        b2_shape = (self.config.n_labels)
        U = tf.get_variable('U', shape=U_shape,
                            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', shape=b2_shape,
                             initializer=tf.constant_initializer())
        pred = tf.matmul(h_dropout, U) + b2
        return pred

    def _agg_outputs(self, outputs, averaging=True):
        if averaging:
            mask = tf.cast(self.mask_placeholder, tf.float32)
            n_words = tf.reduce_sum(mask, axis=1, keep_dims=True)
            mask_stack = tf.stack([mask] * self.config.state_size, axis=-1)
            sum_outputs = tf.reduce_sum(outputs * mask_stack, axis=1)
            h = sum_outputs / n_words
        else:
            mask = tf.cast(self.mask_placeholder, tf.int32)
            n_words = tf.reduce_sum(mask, axis=1)
            n_seq = self.config.max_comment_size
            batch_size = tf.shape(self.input_placeholder)[0]
            ind = tf.range(batch_size) * n_seq + (n_words - 1)
            outputs_flat = tf.reshape(outputs, (-1, self.config.state_size))
            h = tf.gather(outputs_flat, ind)
        return h

    def _add_loss_op(self, pred):
        labels = self.labels_placeholder
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                       logits=pred)
        if hasattr(self.config, 'class_weights'):
            weighted_mean = tf.reduce_mean(loss * self.config.class_weights)
            return weighted_mean / self.config.class_weights_sum
        else:
            return tf.reduce_mean(loss)

    def _add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def _train_on_batch(self, sess, inputs_batch, masks_batch, labels_batch):
        feed = self._create_feed_dict(inputs_batch, masks_batch, labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def _eval_on_batch(self, sess, inputs_batch, masks_batch, labels_batch):
        feed = self._create_feed_dict(inputs_batch, masks_batch, labels_batch)
        loss = sess.run(self.loss, feed_dict=feed)  #
        return loss

    def _predict_on_batch(self, sess, inputs_batch, masks_batch):
        feed = self._create_feed_dict(inputs_batch, masks_batch)
        pred_batch = sess.run(tf.sigmoid(self.pred), feed_dict=feed)  #
        return pred_batch

    def _run_epoch_train(self, sess, inputs, masks, labels, shuffle):
        n_minibatches = len(np.arange(0, len(inputs), self.config.batch_size))
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        minibatches = utils.minibatch(self.config.batch_size,
                                      inputs, labels=labels, masks=masks,
                                      shuffle=shuffle)
        loss = 0
        for i, batch in enumerate(minibatches):
            loss += self._train_on_batch(sess, *batch)
            force = (i + 1) == n_minibatches
            prog.update(i + 1, [('train_loss', loss)], force=force)
        loss /= (i + 1)
        return loss

    def _run_epoch_dev(self, sess, inputs, masks, labels, shuffle):
        minibatches = utils.minibatch(self.config.batch_size,
                                      inputs, labels=labels, masks=masks,
                                      shuffle=shuffle)
        loss = 0
        for i, batch in enumerate(minibatches):
            loss += self._eval_on_batch(sess, *batch)
        loss /= (i + 1)
        print 'dev loss = {:.4f}'.format(loss)
        return loss

    def _transform_inputs(self, tokens):
        return np.array(tokens_to_ids(tokens, self.word2id))

    def train(self, sess,
              tokens_train, masks_train, labels_train,
              tokens_dev=None, masks_dev=None, labels_dev=None,
              shuffle=True):
        inputs_train = self._transform_inputs(tokens_train)
        list_train_loss = []
        if tokens_dev is not None:
            inputs_dev = self._transform_inputs(tokens_dev)
            list_dev_loss = []
        for epoch in range(self.config.n_epochs):
            print "Epoch = {}/{}:".format(epoch + 1, self.config.n_epochs)
            train_loss = self._run_epoch_train(sess,
                                               inputs_train, masks_train,
                                               labels_train, shuffle=shuffle)
            list_train_loss.append(train_loss)
            if tokens_dev is not None:
                dev_loss = self._run_epoch_dev(sess,
                                               inputs_dev, masks_dev,
                                               labels_dev, shuffle=False)
                list_dev_loss.append(dev_loss)
        if tokens_dev is not None:
            return list_train_loss, list_dev_loss
        else:
            return list_train_loss

    def predict(self, sess, tokens, masks):
        inputs = self._transform_inputs(tokens)
        minibatches = utils.minibatch(self.config.batch_size,
                                      inputs, labels=None, masks=masks,
                                      shuffle=False)
        list_y_score = []
        for i, batch in enumerate(minibatches):
            inputs_batch, masks_batch = batch
            score = self._predict_on_batch(sess, inputs_batch, masks_batch)
            list_y_score.append(score)
        y_score = np.vstack(list_y_score)
        return y_score
