import os
import numpy as np
import tensorflow as tf
import logging
import yaml
from model import Model
import preprocess
import utils
import evaluate


# Example config for reference
example_config = {'exp_name': 'ff_full_1',
                  'n_epochs': 500,
                  'n_features': 300,
                  'n_labels': 6,
                  'class_weights': np.array([1.0, 9.58871448, 1.810155,
                                             31.99581504, 1.94160208,
                                             10.88540896]),
                  'n_layers': 2,
                  'hidden_sizes': [20, 20],  # int or list of int
                  'lr': .0005,
                  'batch_size': 2048,
                  'dropout': True,
                  'keep_prob': 0.6,
                  'activation': tf.nn.relu,
                  'optimizer': tf.train.AdamOptimizer
                  }


class FeedForwardNeuralNetwork(Model):

    def _add_placeholders(self):
        input_shape = (None, self.config.n_features)
        labels_shape = (None, self.config.n_labels)
        dropout_shape = ()
        self.input_placeholder = tf.placeholder(tf.float32, input_shape)
        self.labels_placeholder = tf.placeholder(tf.float32, labels_shape)
        self.dropout_placeholder = tf.placeholder(tf.float32, dropout_shape)

    def _create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        feed_dict = {self.input_placeholder: inputs_batch,
                     self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def _add_prediction_op(self):
        try:
            sizes = list(self.config.hidden_sizes)
        except TypeError:
            sizes = [self.config.hidden_sizes] * self.config.n_layers
        assert len(sizes) == self.config.n_layers
        sizes = [self.config.n_features] + sizes + [self.config.n_labels]

        activation = self.config.activation
        initializer = tf.contrib.layers.xavier_initializer()

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
                    if self.config.dropout:
                        z = tf.nn.dropout(z, self.dropout_placeholder)
                else:
                    z = tf.matmul(h, W) + b
                    h = activation(z)
        pred = z
        return pred

    def _add_loss_op(self, pred):
        labels = self.labels_placeholder
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                       logits=pred)
        loss = (tf.reduce_mean(loss * self.config.class_weights) /
                self.config.mean_class_weights)
        return loss

    def _add_training_op(self, loss):
        opt = self.config.optimizer(learning_rate=self.config.lr)
        train_op = opt.minimize(loss)
        return train_op

    def _train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self._create_feed_dict(inputs_batch, labels_batch,
                                      dropout=self.config.keep_prob)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def _loss_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self._create_feed_dict(inputs_batch, labels_batch)
        loss = sess.run(self.loss, feed_dict=feed)
        return loss

    def _predict_on_batch(self, sess, inputs_batch):
        feed = self._create_feed_dict(inputs_batch)
        prob = sess.run(tf.sigmoid(self.pred), feed_dict=feed)
        return prob

    def _run_epoch_train(self, sess, inputs, labels, shuffle):
        n_minibatches = len(np.arange(0, len(inputs), self.config.batch_size))
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        minibatches = utils.minibatch(self.config.batch_size,
                                      inputs, labels=labels, shuffle=shuffle)
        mean_loss = 0
        for i, (inputs_batch, labels_batch) in enumerate(minibatches):
            loss = self._train_on_batch(sess, inputs_batch, labels_batch)
            mean_loss += loss
            force = (i + 1) == n_minibatches
            prog.update(i + 1, [('train_loss', loss)], force=force)
        mean_loss /= (i + 1)
        return mean_loss

    def _run_epoch_dev(self, sess, inputs, labels, shuffle):
        minibatches = utils.minibatch(self.config.batch_size,
                                      inputs, labels=labels, shuffle=shuffle)
        mean_loss = 0
        for i, (inputs_batch, labels_batch) in enumerate(minibatches):
            loss = self._loss_on_batch(sess, inputs_batch, labels_batch)
            mean_loss += loss
        mean_loss /= (i + 1)
        return mean_loss

    def _run_epoch_pred(self, sess, inputs):
        minibatches = utils.minibatch(self.config.batch_size,
                                      inputs, labels=None, shuffle=False)
        list_y_prob = []
        for i, inputs_batch in enumerate(minibatches):
            prob = self._predict_on_batch(sess, inputs_batch)
            list_y_prob.append(prob)
        y_prob = np.vstack(list_y_prob)
        return y_prob

    def _run_epoch_eval(self, sess, y_prob, labels, metric):
        score = evaluate.evaluate(labels, y_prob, metric=metric, average=True)
        return score

    def _transform_inputs(self, tokens):
        inds = preprocess.tokens_to_ids(tokens, self.word2id)
        inputs = preprocess.average_sentence_vectors(inds, self.emb_matrix)
        return inputs

    def train(self, sess, tokens_train, labels_train, tokens_dev, labels_dev,
              metric='roc', saver=None, save_prefix=None, shuffle=True):
        logger = logging.getLogger()
        logging.basicConfig(format='%(asctime)s -- %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
        inputs_train = self._transform_inputs(tokens_train)
        inputs_dev = self._transform_inputs(tokens_dev)
        list_loss_train = []
        list_loss_dev = []
        list_score_train = []
        list_score_dev = []
        best_score_dev = 0
        best_y_prob_train = None
        best_y_prob_dev = None
        for epoch in range(self.config.n_epochs):
            logger.info("")
            logger.info("Epoch = {}/{}:".format(epoch+1, self.config.n_epochs))
            loss_train = self._run_epoch_train(sess,
                                               inputs_train, labels_train,
                                               shuffle=shuffle)
            loss_dev = self._run_epoch_dev(sess,
                                           inputs_dev, labels_dev,
                                           shuffle=False)
            y_prob_train = self._run_epoch_pred(sess, inputs_train)
            y_prob_dev = self._run_epoch_pred(sess, inputs_dev)
            score_train = self._run_epoch_eval(sess,
                                               y_prob_train, labels_train,
                                               metric=metric)
            score_dev = self._run_epoch_eval(sess,
                                             y_prob_dev, labels_dev,
                                             metric=metric)
            metric_name = evaluate.metric_long[metric]
            logger.info("train loss = {:.4f}".format(loss_train))
            logger.info("dev loss = {:.4f}".format(loss_dev))
            logger.info("train {} = {:.4f}".format(metric_name, score_train))
            logger.info("dev {} = {:.4f}".format(metric_name, score_dev))
            list_loss_train.append(loss_train)
            list_loss_dev.append(loss_dev)
            list_score_train.append(score_train)
            list_score_dev.append(score_dev)
            if score_dev > best_score_dev:
                best_score_dev = score_dev
                best_y_prob_train = y_prob_train
                best_y_prob_dev = y_prob_dev
                logger.info("New best dev {} = {:.4f}".format(metric_name,
                                                              score_dev))
                if saver:
                    logger.info("Saving new best model...")
                    saver.save(sess, save_prefix+'.weights')
        return (list_loss_train, list_loss_dev,
                list_score_train, list_score_dev,
                best_y_prob_train, best_y_prob_dev)

    def predict(self, sess, tokens):
        inputs = self._transform_inputs(tokens)
        y_prob = self._run_epoch_pred(sess, inputs)
        return y_prob


# Module method
def load_and_process(train_data_file, test_data_file=None,
                     train_tokens_file=None, test_tokens_file=None,
                     embed_size=300, label_names=None,
                     fraction_dev=0.3, debug=False):
    # Get glove/w2v data
    emb_data = preprocess.get_glove(embed_size)

    # Load and (optionally) subset train data
    train_data = preprocess.load_data(train_data_file, debug=debug)

    # Load test data
    if test_data_file:
        test_data = preprocess.load_data(test_data_file, debug=debug)
        id_test = test_data['id']

    # Tokenize train comments or load pre-tokenized train comments
    if debug or (train_tokens_file is None):
        tokens = preprocess.tokenize_df(train_data)
    else:
        tokens = preprocess.load_tokenized_comments(train_tokens_file)

    # Tokenize test comments or load pre-tokenized test comments
    if test_data_file:
        if test_tokens_file is None:
            tokens_test = preprocess.tokenize_df(test_data)
        else:
            tokens_test = preprocess.load_tokenized_comments(test_tokens_file)

    # Load train labels
    if label_names is None:
        label_names = ['toxic', 'severe_toxic', 'obscene',
                       'threat', 'insult', 'identity_hate']
    labels = preprocess.filter_labels(train_data, label_names)

    # Split to train and dev sets
    train_dev_set = preprocess.split_train_dev(tokens, labels,
                                               fraction_dev=fraction_dev)
    if test_data_file:
        test_set = (id_test, tokens_test)
    else:
        test_set = None

    return emb_data, train_dev_set, test_set


# Module method
def run(config, emb_data, train_dev_set, test_set=None,
        label_names=None, out_dir='./', debug=False):
    # Create save directory
    save_dir = os.path.join(out_dir, config['exp_name'])
    if debug:
        save_dir += '_debug'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_prefix = os.path.join(save_dir, config['exp_name'])

    # Write config to log file
    with open(save_prefix+'.log', 'w') as f:
        yaml.dump(config, f)

    # Create file handler for logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(save_prefix+'.log')
    ch = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s -- %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Initializing
    logger.info("")
    logger.info("Initializing...")
    (tokens_train, labels_train, tokens_dev, labels_dev) = train_dev_set
    if test_set:
        (id_test, tokens_test) = test_set
    if label_names is None:
        label_names = ['toxic', 'severe_toxic', 'obscene',
                       'threat', 'insult', 'identity_hate']

    # Fit
    logger.info("Training...")
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        logger.info("Building model...")
        obj = FeedForwardNeuralNetwork(config, emb_data=emb_data)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        logger.info("Training model...")
        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            results = obj.train(sess,
                                tokens_train, labels_train,
                                tokens_dev, labels_dev,
                                saver=saver, save_prefix=save_prefix)
            (list_loss_train, list_loss_dev,
             list_score_train, list_score_dev,
             y_prob_train, y_prob_dev) = results

    # Predict test set
    logger.info("")
    logger.info("Testing...")
    if test_set:
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            logger.info("Rebuilding model...")
            obj = FeedForwardNeuralNetwork(config, emb_data=emb_data)
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            logger.info("Restoring best model...")
            with tf.Session(graph=graph) as sess:
                sess.run(init_op)
                saver.restore(sess, save_prefix+'.weights')
                logger.info("Predicting labels for test set...")
                y_prob_test = obj.predict(sess, tokens_test)
        # Save y_prob_test to csv
        logger.info("Saving test prediction...")
        y_prob_test_df = utils.y_prob_to_df(y_prob_test, id_test, label_names)
        y_prob_test_df.to_csv(save_prefix+'_test.csv', index=False)

    # Evaluate and plot
    logger.info("")
    logger.info("Evaluating...")
    evaluate.plot_loss(list_loss_train, list_loss_dev,
                       save_prefix=save_prefix)
    evaluate.plot_score(list_score_train, list_score_dev,
                        save_prefix=save_prefix)
    y_dict = {'train': (labels_train, y_prob_train),
              'dev': (labels_dev, y_prob_dev)}
    evaluate.evaluate_full(y_dict, metric='roc', names=label_names,
                           plot=True, save_prefix=save_prefix)
    evaluate.evaluate_full(y_dict, metric='prc', names=label_names,
                           plot=True, save_prefix=save_prefix)
