import utils
import os
import pickle

from tensorflow.contrib.rnn import MultiRNNCell, BasicLSTMCell, GRUCell
import tensorflow as tf
tf.reset_default_graph()


class Model(object):

    def __init__(self, config):

        self.config = config

        with tf.device('/cpu:0'):

            self.inputs = tf.placeholder(tf.float32,
                                         shape=[None, None, self.config.n_features],
                                         name='inputs')
            self.labels = tf.sparse_placeholder(tf.int32,
                                                name='labels')

            sequence_lengths = utils.compute_seq_lengths(self.inputs)

            batch_size = self.config.batch_size

            self.conv_weights = {}
            self.conv_biases = {}

            # TODO: Add batch normalization to RNN

            conv_output = tf.reshape(self.inputs, [batch_size, self.config.input_max_len, self.config.n_features, 1])
            layer_output = 1

            with tf.variable_scope("CNN"):

                for i in range(self.config.num_conv_layers):

                    self.conv_weights['W_conv' + str(i+1)] = tf.Variable(tf.random_normal([5, 5, layer_output, 32*(i+1)]), name=('W_conv' + str(i+1)))
                    self.conv_biases['b_conv' + str(i+1)] = tf.Variable(tf.random_normal([32*(i+1)]), name=('b_conv' + str(i+1)))
                    layer_output = 32 * (i + 1)

                    conv_output = tf.nn.conv2d(conv_output, self.conv_weights['W_conv' + str(i+1)], strides=[1, 1, 1, 1], padding='SAME')
                    conv_output = tf.layers.batch_normalization(conv_output)

            conv_output = tf.reshape(conv_output, [batch_size, self.config.input_max_len, -1])

            def rnn_cell():

                if self.config.rnn_type == 'lstm':
                    return BasicLSTMCell(self.config.rnn_size)
                elif self.config.rnn_type == 'gru':
                    return GRUCell(self.config.rnn_size)
                else:
                    raise Exception('Invalid rnn type: {} (Must be lstm or gru)'.format(self.config.rnn_type))

            with tf.variable_scope('RNN'):
                if self.config.rnn_layers == 1:
                    self.fw_rnn_cell = rnn_cell()
                    if self.config.bidirectional_rnn:
                        self.bw_rnn_cell = rnn_cell()
                else:
                    self.fw_rnn_cell = MultiRNNCell([rnn_cell() for _ in range(self.config.rnn_layers)])
                    if self.config.bidirectional_rnn:
                        self.bw_rnn_cell = MultiRNNCell([rnn_cell() for _ in range(self.config.rnn_layers)])

                self.fw_rnn_state = self.fw_rnn_cell.zero_state(batch_size, dtype=tf.float32)
                if self.config.bidirectional_rnn:
                    self.bw_rnn_state = self.bw_rnn_cell.zero_state(batch_size, dtype=tf.float32)
                else:
                    self.initial_rnn_state = self.fw_rnn_state

                if self.config.bidirectional_rnn:
                    rnn_outputs, self.rnn_state = tf.nn.bidirectional_dynamic_rnn(self.fw_rnn_cell, self.bw_rnn_cell, conv_output,
                                                                         sequence_lengths, initial_state_fw=self.fw_rnn_state,
                                                                         initial_state_bw=self.bw_rnn_state, dtype=tf.float32)
                else:
                    rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.fw_rnn_cell, conv_output,
                                                           sequence_lengths, initial_state=self.fw_rnn_state, dtype=tf.float32)

            outputs = tf.reshape(rnn_outputs, [-1, self.config.rnn_size])

            if (not self.config.bidirectional_rnn) and self.config.future_context > 0:
                with tf.variable_scope("Lookahead"):
                    # TODO: Make much faster (tf.map_fn)
                    self.row_conv_weights = tf.Variable(tf.random_normal([1, self.config.rnn_size, self.config.future_context+1]),
                                                        name='W_row_conv')
                    # outputs = tf.concat([outputs, np.zeros([self.config.future_context, self.config.rnn_size])], axis=0)
                    outputs = tf.reshape(outputs, [batch_size, -1, self.config.rnn_size])
                    row_conv_output = tf.nn.conv1d(outputs, self.row_conv_weights, stride=1, padding='SAME')
                    print(row_conv_output)

                    # outputs = tf.map_fn(
                    #     lambda t: tf.map_fn(
                    #         lambda i: tf.reduce_sum(tf.map_fn(
                    #             lambda j: tf.multiply(self.row_conv_weights[i, j], outputs[t+j, i]),
                    #             tf.range(self.config.future_context), dtype=tf.float32
                    #         )),
                    #         tf.range(self.config.rnn_size), dtype=tf.float32
                    #     ),
                    #     tf.range(outputs.shape[0] - self.config.future_context), dtype=tf.float32
                    # )

            with tf.variable_scope("Fully_Connected"):

                fc_W = tf.Variable(tf.truncated_normal([self.config.rnn_size, self.config.n_classes], stddev=0.1), name='W_fc')
                fc_b = tf.Variable(tf.constant(0., shape=[self.config.n_classes]), name='b_fc')

                logits = tf.matmul(outputs, fc_W) + fc_b

                # logits = tf.contrib.layers.fully_connected(rnn_outputs, self.config.n_classes)

                logits = tf.reshape(logits, [batch_size, -1, self.config.n_classes])
                logits = tf.transpose(logits, (1, 0, 2))

        with tf.device('/gpu:0'):

            loss = tf.nn.ctc_loss(self.labels, logits, sequence_lengths)
            self.cost = tf.reduce_mean(loss)

            cost_summary = tf.summary.scalar('cost', self.cost)
            self.summary = tf.summary.merge([cost_summary])

            if self.config.optimizer == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.config.learning_rate).minimize(self.cost)
            elif self.config.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.config.learning_rate).minimize(self.cost)
            else:
                raise Exception('Invalid optimizer: {}'.format(self.config.optimizer))

            if self.config.beam_width > 0:
                self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, sequence_lengths, beam_width=self.config.beam_width)
            else:
                self.decoded, _ = tf.nn.ctc_greedy_decoder(logits, sequence_lengths)

            self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))
            ler_summary = tf.summary.scalar('label error rate', self.ler)
            self.summary = tf.summary.merge([ler_summary])

        self.saver = tf.train.Saver()

    @classmethod
    def load(cls, params_filepath, checkpoint_filepath, sess):

        with open(params_filepath, 'rb') as f:
            hparams = pickle.load(f)

        obj = cls(hparams)
        obj.saver.restore(sess, checkpoint_filepath)

        return obj

    def train(self, inputs, targets, sess):

        return sess.run([self.cost, self.optimizer, self.summary], feed_dict={
            self.inputs: inputs,
            self.labels: targets
        })

    def eval(self, inputs, targets, sess):

        return sess.run([self.ler, self.summary], feed_dict={
            self.inputs: inputs,
            self.labels: targets
        })

    def start_stream(self, sess):

        assert not self.config.bidirectional_rnn

        self.initial_rnn_state = self.fw_rnn_state
        sess.run([self.initial_rnn_state])

    def streaming_infer(self, inputs, sess):

        decoded, rnn_state = sess.run([self.decoded, self.rnn_state], feed_dict={
            self.inputs: inputs
        })

        self.fw_rnn_state = rnn_state
        return decoded

    def end_stream(self, sess):

        self.fw_rnn_state = self.initial_rnn_state
        sess.run([self.fw_rnn_state])

    def infer(self, inputs, sess):

        return sess.run([self.decoded], feed_dict={
            self.inputs: inputs,
        })

    def save(self, dir, sess, global_step=None):

        with open(os.path.join(dir, 'hparams'), 'wb') as f:
            pickle.dump(self.config, f)

        self.saver.save(sess, os.path.join(dir, 'checkpoints', 'checkpoint'), global_step=global_step)