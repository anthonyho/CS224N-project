import tensorflow as tf
from preprocess import get_glove
from model import Model

config = {
    'batch_size' : 20,
    'state_size' : 50,
    'max_comment_size'  : 100,

}

class RNNModel(Model):

    def __init__(self, config=None, emb_data=None, glove_dim=None):
        # Load word embedding data from memory if already loaded
        if emb_data is not None:
            self.emb_matrix = emb_data[0]
            self.word2id = emb_data[1]
            self.id2word = emb_data[2]
        # Load glove data from file
        elif glove_dim is not None:
            self.emb_matrix, self.word2id, self.id2word = get_glove(glove_dim)
        # Load config and build
        self.config = config
        self.build()

    def _add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, 
            [self.config['batch_size'], self.config['max_comment_size'],1])
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
        self.mask_placeholder = tf.placeholder(tf.bool,shape=(None,self.config['max_comment_size']))

    def _create_feed_dict(self, inputs_batch, masks_batch, labels_batch=None):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.mask_placeholder : masks_batch,
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def _add_prediction_op(self):
        embed_matrix = tf.Variable(initial_value=self.emb_matrix)      
        embeddings2 = tf.nn.embedding_lookup(embed_matrix, self.input_placeholder)
        embeddings = tf.reshape(embeddings2,(-1,Config.max_length,Config.n_features*Config.embed_size))
        
        x = embeddings
        cell = tf.contrib.rnn.BasicRNNCell(self.config['state_size'])
        out, state = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
        preds = tf.sigmoid(state)
    

    def _add_loss_op(self, preds):
        loss = tf.reduce_mean(tf.boolean_mask(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=preds,labels=self.labels_placeholder),
                self.mask_placeholder))

    def _add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def _train_on_batch(self, sess, inputs_batch, masks_batch, labels_batch):
        feed = self._create_feed_dict(inputs_batch, masks_batch, labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def _run_epoch(self, sess, inputs, masks, labels, shuffle):
        minibatches = utils.minibatch(self.config['batch_size'],
                                      inputs, labels, masks, shuffle)
        loss = 0
        for i, (inputs_batch, masks_batch, labels_batch) in enumerate(minibatches):
            loss += self._train_on_batch(sess, inputs_batch, masks_batch, labels_batch)
        loss /= (i + 1)
        return loss

    def train(self, sess, inputs, masks, labels, shuffle=True):
        list_loss = []
        for epoch in range(self.config['n_epochs']):
            list_loss.append(self._run_epoch(sess, inputs, masks, labels, shuffle))
        return list_loss

    #def predict(self, sess, tokens):
    #    feed = self._create_feed_dict(inputs)
    #    pred = sess.run(self.pred, feed_dict=feed)
    #    y_score = utils.sigmoid(pred)
    #    return y_score

    def save_weights(self, file_path):
        raise NotImplementedError()

    def load_weights(self, file_path):
        raise NotImplementedError()

    def build(self):
        self._add_placeholders()
        self.pred = self._add_prediction_op()
        self.loss = self._add_loss_op(self.pred)
        self.train_op = self._add_training_op(self.loss)
