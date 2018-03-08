import tensorflow as tf
from preprocess import get_glove


class Config(object):

    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)


class Model(object):

    def __init__(self, config=None, emb_data=None, glove_dim=None):
        # Load word embedding data from memory if already loaded
        if emb_data is not None:
            self.emb_matrix = emb_data[0].astype('float32')
            self.word2id = emb_data[1]
            self.id2word = emb_data[2]
        # Load glove data from file
        elif glove_dim is not None:
            self.emb_matrix, self.word2id, self.id2word = get_glove(glove_dim)
            self.emb_matrix = self.emb_matrix.astype('float32')
        # Load config and build
        self.config = Config(config)
        self.build()

    def _add_placeholders(self):
        raise NotImplementedError()

    def _create_feed_dict(self, inputs_batch, labels_batch=None):
        raise NotImplementedError()

    def _add_prediction_op(self):
        raise NotImplementedError()

    def _add_loss_op(self, pred):
        raise NotImplementedError()

    def _add_training_op(self, loss):
        raise NotImplementedError()

    def _train_on_batch(self, sess, inputs_batch, labels_batch):
        raise NotImplementedError()

    def _run_epoch(self, sess, inputs, labels):
        raise NotImplementedError()

    def _transform_inputs(self, tokens):
        raise NotImplementedError()

    def train(self, tokens, labels):
        raise NotImplementedError()

    def predict(self, tokens):
        # saving self.predicted_labels
        raise NotImplementedError()

    def save_weights(self, file_path):
        raise NotImplementedError()

    def load_weights(self, file_path):
        raise NotImplementedError()

    def build(self):
        self._add_placeholders()
        self.pred = self._add_prediction_op()
        self.loss = self._add_loss_op(self.pred)
        self.train_op = self._add_training_op(self.loss)
