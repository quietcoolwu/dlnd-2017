#!/usr/bin/env python

import math
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class TwoLayerDNNReLU(object):
    def __init__(self):
        """
        A Deep Neural Network with 1 hidden layer(ReLU)
        """
        # Parameters
        self.learning_rate = 0.001
        self.training_epochs = 20
        self.batch_size = 128

        # 28 x 28 = 784 pixels
        self.n_input = 784
        # 0-9: 10 labels
        self.n_classes = 10

        # Only one hidden-layer here with 256 neurons
        self.n_hidden_layer = 256
        self.save_file = './tmp/'
        # Import MNIST data
        if not os.path.exists(self.save_file):
            os.makedirs(self.save_file)
        self.data = input_data.read_data_sets(
            os.path.join(self.save_file, 'datasets'),
            one_hot=True,
            reshape=False)
        # Remove previous Tensors and Operations
        tf.reset_default_graph()
        print('Static Init Done!')

    def model_initialize(self):
        model_init_flag = False
        weights = {
            'hidden_layer':
            tf.Variable(tf.random_normal([self.n_input, self.n_hidden_layer])),
            'out':
            tf.Variable(
                tf.random_normal([self.n_hidden_layer, self.n_classes]))
        }

        biases = {
            'hidden_layer':
            tf.Variable(tf.random_normal([self.n_hidden_layer])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # tf Graph input
        self.features = tf.placeholder("float", [None, 28, 28, 1])
        self.labels = tf.placeholder("float", [None, self.n_classes])
        # like np.reshape
        features_flat = tf.reshape(self.features, [-1, self.n_input])

        # Hidden layer with RELU activation
        hidden_layer_in = tf.add(
            tf.matmul(features_flat, weights['hidden_layer']),
            biases['hidden_layer'])
        hidden_layer_out = tf.nn.relu(hidden_layer_in)
        # Output layer with linear activation
        logits = tf.add(
            tf.matmul(hidden_layer_out, weights['out']), biases['out'])

        # Define loss and optimizer
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels))
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(cost)

        model_init_flag = not model_init_flag
        return model_init_flag

    def model_train(self):
        train_flag = False
        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for _ in range(self.training_epochs):
                total_batch = math.ceil(self.data.train.num_examples /
                                        self.batch_size)
                # Loop over all batches
                for _ in range(total_batch):
                    # next_batch: like an iterator
                    batch_x, batch_y = self.data.train.next_batch(
                        self.batch_size)
                    """
                    Run optimization (backprop) and cost (to get loss value)
                    optimizer -> cost -> labels, logits
                    -> ReLU * w + b-> x_flat * w + b -> input
                    """
                    sess.run(
                        self.optimizer,
                        feed_dict={
                            self.features: batch_x,
                            self.labels: batch_y
                        })
                # batch_x: (128, 28, 28, 1) gray-scale; batch_y: (128, 10)
                print(batch_x.shape, batch_y.shape)

        print('Training ended.')
        train_flag = not train_flag
        return train_flag

    @staticmethod
    def run():
        case = TwoLayerDNNReLU()
        model_init = case.model_initialize()
        assert model_init
        model_train_save = case.model_train()
        assert model_train_save


if __name__ == '__main__':
    TwoLayerDNNReLU.run()
