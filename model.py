import tensorflow as tf
from vocab import get_glove


class Model(object):

    def __init__(self, config=None, emb_data=None, glove_dim=None):
        # Load word embedding data from memory if already loaded
        if emb_data is not None:
            self.emb_matrix = emb_data[0]
            self.word2id = emb_data[1]
            self.id2word = emb_data[2]
        # Load glove data from file 
        elif glove_dim is not None:
            glove_prefix = 'data/glove/glove.6B.'
            glove_suffix = 'd.txt'
            glove_file = glove_prefix+str(glove_dim)+glove_suffix
            (self.emb_matrix, self.word2id,
             self.id2word) = get_glove(glove_file, glove_dim)
        # Load config and build
        self.config = config
        self.build()

    def add_placeholders(self):
        raise NotImplementedError()
    
    def create_feed_dict(self):
        raise NotImplementedError()

    def add_prediction_op(self):
        raise NotImplementedError()

    def add_loss_op(self):
        raise NotImplementedError()

    def add_training_op(self):
        raise NotImplementedError()

    def train(self, list_tokens, array_labels):
        raise NotImplementedError()

    def save_weights(self, file_path):
        raise NotImplementedError()

    def predict(self, list_list_tokens):
        # saving self.predicted_labels
        raise NotImplementedError()

    def load_weights(self, file_path):
        raise NotImplementedError()

    def evaluate(self, array_true_labels, method='roc'):
        # calling self.predicted_labels
        raise NotImplementedError()

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
