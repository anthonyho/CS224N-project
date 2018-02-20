from vocab import get_glove
import tensorflow as tf

class Model(object):

    def __init__(self, glove_dim, config=None):
        glove_prefix = 'data/glove/glove.6B.'
        glove_suffix = 'd.txt'
        glove_file = glove_prefix+str(glove_dim)+glove_suffix
        self.emb_matrix, self.word2id, self.id2word = get_glove(glove_file, glove_dim)
        self.config = config
        self.build()

    def build(self, hyperparams_dict=None):
        raise NotImplementedError()

    def train(self, list_tokens, array_labels):
        raise NotImplementedError()

    def save_weights(self, file_path):
        raise NotImplementedError()

    def predict(self, list_tokens):
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
