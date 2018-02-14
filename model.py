from vocab import get_glove

class Model(object):

    def __init__(self, glove_dim):
        glove_prefix = 'data/glove/glove.6B.'
        glove_suffix = 'd.txt'
        glove_file = glove_prefix+str(glove_dim)+glove_suffix
        self.emb_matrix, self.word2id, self.id2word = get_glove(glove_file, glove_dim)

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
