import os
import tensorflow as tf
import numpy as np
from nltk.tokenize.moses import MosesDetokenizer
import tables

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# normc weight initializer from CS294-112
def normc_initializer(std=1.0):
    def _initializer(shape):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class FLEX(object):

    def __init__(self, params):

        self.img_embed_size = params['img_embed_size']
        self.embedding_size = params['lstm_embedding_size']
        self.num_hidden_lstm = params['num_hidden_lstm']
        self.vocab_size = params['vocab_size']
        self.max_length = params['max_length']
        self.learning_rate = params['learning_rate']
        self.dropout = params['dropout']
        self.lambda_value = params['lambda_value']
        self.img_feature_size = params['img_feature_size']

        tf.reset_default_graph()

        # -------- Placeholders --------
        self.img_feats = tf.placeholder(tf.float32, [None, self.img_feature_size])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.seq_lengths = tf.placeholder(tf.int32, [None])
        self.rnn_inputs = tf.placeholder(tf.int32, [None, self.max_length])
        self.rnn_outputs_input = tf.placeholder(tf.int32, [None, self.max_length])
        self.c_state1 = tf.placeholder(tf.float32, [None, self.num_hidden_lstm])
        self.h_state1 = tf.placeholder(tf.float32, [None, self.num_hidden_lstm])
        self.c_state2 = tf.placeholder(tf.float32, [None, self.num_hidden_lstm])
        self.h_state2 = tf.placeholder(tf.float32, [None, self.num_hidden_lstm])
        self.learning_rate_input = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.scores_input = tf.placeholder(tf.float32, [None, self.max_length, self.vocab_size])

        self.initial_state2 = tf.contrib.rnn.LSTMStateTuple(self.c_state2, self.h_state2)
        self.initial_state1 = tf.contrib.rnn.LSTMStateTuple(self.c_state1, self.h_state1)

        # -------- Convert lstm output indices to one-hot vectors --------
        self.lstm_outputs = tf.one_hot(self.rnn_outputs_input, depth=self.vocab_size)

        # -------- Initialize word embedding --------
        self.w_embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
        self.embed_word = tf.nn.embedding_lookup(self.w_embedding, self.rnn_inputs)

        # Initialize image feature embedding
        self.img_embedding = tf.Variable(normc_initializer(1.0)([self.img_feature_size, self.img_embed_size]))
        self.img_embedding_bias = tf.Variable(tf.constant_initializer(0.0)((self.img_embed_size,)), name="img_embed_bias")

        self.embed_visual_feats = tf.matmul(self.img_feats, self.img_embedding) + self.img_embedding_bias

        # -------- LSTM 1 --------
        with tf.variable_scope('rnn1'):
            if self.dropout:
                self.lstm_cell1 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_hidden_lstm, state_is_tuple=True), output_keep_prob=self.keep_prob)
            else:
                self.lstm_cell1 = tf.contrib.rnn.LSTMCell(self.num_hidden_lstm, state_is_tuple=True)
            self.zero_initial_state1 = self.lstm_cell1.zero_state(self.batch_size, tf.float32)
            self.lstm_outputs1, self.lstm_states1 = tf.nn.dynamic_rnn(self.lstm_cell1, self.embed_word, dtype=tf.float32, sequence_length=self.seq_lengths,
                                                                      initial_state=self.initial_state1)

        # -------- LSTM2 embedding variables --------
        self.lstm2_embed_w = tf.Variable(normc_initializer(1.0)(
            [self.num_hidden_lstm + int(self.embed_visual_feats.get_shape()[-1]),
             self.num_hidden_lstm]))
        self.lstm2_embed_b = tf.Variable(tf.constant_initializer(0.0)((self.num_hidden_lstm,)), name="lstm_2_embed_bias")

        self.hidden_projection = lambda x: tf.matmul(tf.concat([x, self.embed_visual_feats], axis=-1), self.lstm2_embed_w) + self.lstm2_embed_b

        # -------- Embed concatenation of visual feats and the output of LSTM1 for feed to LSTM2 --------
        self.lstm_outputs1 = tf.transpose(self.lstm_outputs1, [1, 0, 2])
        self.lstm_inputs2 = tf.map_fn(self.hidden_projection, self.lstm_outputs1)
        self.lstm_inputs2 = tf.transpose(self.lstm_inputs2, [1, 0, 2])

        # -------- LSTM 2 --------
        with tf.variable_scope('rnn2'):
            if self.dropout:
                self.lstm_cell2 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_hidden_lstm, state_is_tuple=True), output_keep_prob=self.keep_prob)
            else:
                self.lstm_cell2 = tf.contrib.rnn.LSTMCell(self.num_hidden_lstm, state_is_tuple=True)
            self.zero_initial_state2 = self.lstm_cell2.zero_state(self.batch_size, tf.float32)
            self.lstm_outputs2, self.lstm_states2 = tf.nn.dynamic_rnn(self.lstm_cell2, self.lstm_inputs2, dtype=tf.float32, sequence_length=self.seq_lengths,
                                                                      initial_state=self.initial_state2)

        # -------- Output layer projection variables --------
        self.output_w = tf.Variable(normc_initializer(1.0)(
            [self.num_hidden_lstm, self.vocab_size]))
        self.output_b = tf.Variable(tf.constant_initializer(0.0)((self.vocab_size,)), name="output_bias")

        self.logit_projection = lambda x: tf.matmul(x, self.output_w) + self.output_b

        # -------- Project output of LSTM to output layer --------
        self.lstm_outputs2 = tf.transpose(self.lstm_outputs2, [1, 0, 2])
        self.final_logits = tf.map_fn(self.logit_projection, self.lstm_outputs2)
        self.final_logits = tf.transpose(self.final_logits, [1, 0, 2])
        self.sentence_word_probs = tf.nn.softmax(self.final_logits, dim=-1)

        # -------- Calculate relevance score --------
        score = tf.multiply(self.sentence_word_probs, self.scores_input)
        score = tf.reduce_max(score, 2)

        # -------- Calculate cross entropy loss --------
        self.cross_entropy_loss = self.calculate_cross_entropy(score)

        # -------- Calculate final loss --------
        self.final_loss = tf.reduce_mean(self.cross_entropy_loss)

        # -------- Optimizer --------
        self.optimize_step = tf.train.AdamOptimizer(self.learning_rate_input).minimize(self.final_loss)

        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def calculate_cross_entropy(self, relevance_score):  # A function from sbarratt's interpnet code
        ones = tf.ones((self.batch_size, self.max_length), dtype=tf.int32)
        zeros = tf.zeros((self.batch_size, self.max_length), dtype=tf.int32)
        lengths_transposed = tf.reshape(self.seq_lengths, [-1, 1])
        lengths_transposed = tf.tile(lengths_transposed, [1, self.max_length])
        range_ = tf.range(0, self.max_length, 1)
        range_row = tf.reshape(range_, [-1, 1])
        range_row = tf.transpose(tf.tile(range_row, [1, self.batch_size]))
        mask_int = tf.less(range_row, lengths_transposed)
        mask = tf.where(mask_int, ones, zeros)
        cross_entropy = self.lstm_outputs * tf.log(self.sentence_word_probs + 1e-8)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        cross_entropy = cross_entropy * tf.cast(mask, tf.float32)
        cross_entropy = tf.reduce_sum(tf.subtract(cross_entropy, tf.multiply(relevance_score, self.lambda_value)), reduction_indices=1)
        cross_entropy = cross_entropy / tf.cast(self.seq_lengths, tf.float32)
        return cross_entropy

    def get_visual_feature_batch(self, filename, start, end, index):
        """
        Retrieve a batch of image features from respective .h5 file
        :param filename:
        :param start: start index of the batch
        :param end:
        :param index:
        :return: feature_batch
        """
        f = tables.open_file(filename, mode='r')
        feature_batch = f.root.data[index[start:end], :]
        return feature_batch

    def save(self, f):
        """
        Save the trained model
        :param f: model path
        :return:
        """
        self.saver.save(self.sess, f)

    def load(self, f):
        """
        Restore the trained model
        :param f: model path
        :return:
        """
        self.saver.restore(self.sess, f)

    def train_epoch(self, visual_feat_file, x_word_seq, y_word_seq, lengths, words_relevance_scores, learning_rate, batch_size, keep_prob):
        """
        Train model for one epoch
        :param visual_feat_file:
        :param x_word_seq:
        :param y_word_seq:
        :param lengths:
        :param words_relevance_scores:
        :param learning_rate:
        :param batch_size:
        :param keep_prob:
        :return:
        """

        epoch_loss = 0.0
        no_iterations = 0

        index = (np.arange(x_word_seq.shape[0]).astype(int))
        np.random.shuffle(index)
        for start, end in zip(range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):
            x_img_batch = self.get_visual_feature_batch(visual_feat_file, start, end, index)
            x_exp_batch = x_word_seq[index[start:end]]
            y_batch = y_word_seq[index[start:end]]
            lengths_batch = lengths[index[start:end]]
            scores_batch_ = words_relevance_scores[index[start:end]]
            scores_batch = np.tile(scores_batch_, self.max_length).reshape(batch_size, self.max_length, self.vocab_size)

            c_initial1, h_initial1 = self.sess.run([self.zero_initial_state1], feed_dict={
                self.batch_size: batch_size
            })[0]

            c_initial2, h_initial2 = self.sess.run([self.zero_initial_state2], feed_dict={
                self.batch_size: batch_size
            })[0]

            feed = {
                self.img_feats: x_img_batch,
                self.rnn_inputs: x_exp_batch,
                self.rnn_outputs_input: y_batch,
                self.seq_lengths: lengths_batch,
                self.learning_rate_input: learning_rate,
                self.c_state1: c_initial1,
                self.h_state1: h_initial1,
                self.c_state2: c_initial2,
                self.h_state2: h_initial2,
                self.scores_input: scores_batch,
                self.batch_size: batch_size,
                self.keep_prob: keep_prob
            }

            no_iterations += 1

            loss, _ = self.sess.run([self.final_loss, self.optimize_step], feed_dict=feed)
            epoch_loss += loss

        return epoch_loss / no_iterations

    def validate_epoch(self, visual_feat_file, x_word_seq, y_word_seq, lengths, words_relevance_scores, batch_size, keep_prob):
        """
        Validate the training model after one epoch
        :param visual_feat_file:
        :param x_word_seq:
        :param y_word_seq:
        :param lengths:
        :param words_relevance_scores:
        :param batch_size:
        :param keep_prob:
        :return:
        """

        c_initial1, h_initial1 = self.sess.run([self.zero_initial_state1], feed_dict={
            self.batch_size: batch_size
        })[0]

        c_initial2, h_initial2 = self.sess.run([self.zero_initial_state2], feed_dict={
            self.batch_size: batch_size
        })[0]

        val_loss = 0.0
        no_iterations = 0.0

        index = (np.arange(x_word_seq.shape[0]).astype(int))
        np.random.shuffle(index)
        for start, end in zip(range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):
            scores_batch_ = words_relevance_scores[index[start:end]]
            scores_batch = np.tile(scores_batch_, self.max_length).reshape(batch_size, self.max_length, self.vocab_size)

            feed = {
                self.img_feats: self.get_visual_feature_batch(visual_feat_file, start, end, index),
                self.rnn_inputs: x_word_seq[index[start:end]],
                self.rnn_outputs_input: y_word_seq[index[start:end]],
                self.seq_lengths: lengths[index[start:end]],
                self.c_state1: c_initial1,
                self.h_state1: h_initial1,
                self.c_state2: c_initial2,
                self.h_state2: h_initial2,
                self.scores_input: scores_batch,
                self.batch_size: batch_size,
                self.keep_prob: keep_prob
            }
            loss = self.sess.run([self.final_loss],
                                 feed_dict=feed)[0]
            val_loss += loss
            no_iterations += 1

        return val_loss/no_iterations

    def get_explanation(self, image, id_to_word, word_to_id):
        """
        Generate explanations using a trained model
        :param image:
        :param id_to_word:
        :param word_to_id:
        :return: de-tokenized word sequence
        """
        c1, h1 = self.sess.run([self.zero_initial_state1], feed_dict={
            self.batch_size: 1
        })[0]
        c2, h2 = self.sess.run([self.zero_initial_state2], feed_dict={
            self.batch_size: 1
        })[0]

        indices = []

        k = 0
        ind = 0
        while 1:
            state1, state2, probs = self.sess.run([self.lstm_states1, self.lstm_states2, self.sentence_word_probs],
                                                  feed_dict={
                                                      self.img_feats: image,
                                                      self.rnn_inputs: np.array([ind] + [0] * (self.max_length - 1), dtype=np.int32)[None],
                                                      self.seq_lengths: np.array([1], dtype=np.int32),
                                                      self.c_state1: c1,
                                                      self.h_state1: h1,
                                                      self.c_state2: c2,
                                                      self.h_state2: h2,
                                                      self.batch_size: 1,
                                                      self.keep_prob: 1
                                                  })
            c1, h1 = state1
            c2, h2 = state2
            ind = np.argmax(probs)
            indices.append(ind)
            if id_to_word.get(ind) == '.':
                break

            k += 1

            if k == self.max_length:
                indices.append(word_to_id.get('.'))
                break

        detokenizer = MosesDetokenizer()

        return detokenizer.detokenize([id_to_word.get(i) for i in indices], return_str=True), indices
