#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def two_layer_network_with_ReLU():
    mnist = input_data.read_data_sets(
        './datasets/', one_hot=True, reshape=False)

    # Parameters
    learning_rate = 0.001
    training_epochs = 20
    batch_size = 128

    # 28 x 28 = 784 pixels
    n_input = 784
    # 0-9: 10 labels
    n_classes = 10

    # Only one hidden-layer here with 256 neurons
    n_hidden_layer = 256

    weights = {
        'hidden_layer':
        tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
        'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
    }

    biases = {
        'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # tf Graph input
    x = tf.placeholder("float", [None, 28, 28, 1])
    y = tf.placeholder("float", [None, n_classes])
    # like np.reshape
    x_flat = tf.reshape(x, [-1, n_input])
    """
    Hidden Layer Start!
    """
    # Hidden layer with RELU activation
    hidden_layer_in = tf.add(
        tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
    hidden_layer_out = tf.nn.relu(hidden_layer_in)
    # Output layer with linear activation
    logits = tf.add(tf.matmul(hidden_layer_out, weights['out']), biases['out'])

    # Define loss and optimizer
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
        .minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for _ in range(training_epochs):
            total_batch = mnist.train.num_examples // batch_size
            # Loop over all batches
            for _ in range(total_batch):
                # next_batch: like an iterator
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization (backprop) and cost (to get loss value)
                """
                optimizer -> cost -> labels, logits
                -> ReLU * w + b-> x_flat * w + b -> input
                """
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # batch_x: (128, 28, 28, 1) gray-scale; batch_y: (128, 10)
            print(batch_x.shape, batch_y.shape)


if __name__ == '__main__':
    two_layer_network_with_ReLU()
