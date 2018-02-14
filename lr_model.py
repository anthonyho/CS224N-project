from model import Model
import tensorflow as tf
import preprocessing

config = {'n_features' : 50,
          'n_classes'  : 2,
          'hidden_size': 20,
          'lr'         : .0005
          }

class LogisticRegressionModel(Model):

    def build(self, config=None):
        self.config = config
        self.input_placeholder = tf.placeholder(tf.float32,shape=(None,self.config['n_features']))
        self.labels_placeholder = tf.placeholder(tf.float32,shape=(None,self.config['n_classes']))

        # prediction op
        W1 = tf.Variable(initial_value=tf.contribs.layers.xavier_initializer((self.config['n_features'],self.config['hidden_size']),uniform=False, seed=None, dtype=tf.float32), dtype=tf.float32)
        b1 = tf.Variable(initial_value=tf.zeros(1,self.config['hidden_size']), dtype=tf.float32)
        
        h = tf.nn.relu(tf.matmul(W1,self.input_placeholder)+b1)

        W2 = tf.Variable(initial_value=tf.contribs.layers.xavier_initializer((self.config['hidden_size'],self.config['n_classes']),uniform=False, seed=None, dtype=tf.float32), dtype=tf.float32)
        b2 = tf.Variable(initial_value=tf.zeros(1,self.config['n_classes']), dtype=tf.float32)
	
        self.pred = tf.matmul(W2,h)+b2
        # loss op

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder),logits=self.pred)
        # training op
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config['lr'])
        self.train_op = optimizer.minimize(self.loss)

    def train(self, list_list_tokens, array_labels):
        sess = tf.Session()
        list_list_inds = preprocessing.token_list_to_ids(list_list_tokens,self.word2id)
        sentence_avgs = preprocessing.average_sentence_vectors(list_list_inds,self.emb_matrix):
        feed = {self.input_placeholder : sentence_avgs,
                self.labels_placeholder: array_labels}
        sess.run([self.train_op,self.loss],feed_dict=feed)
