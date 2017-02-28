#!/usr/bin/env python

import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class MNISTDNN(object):
    def __init__():
        # Hyperparameters setting
        self.learning_rate = 0.001
        self.n_input = 784
        self.n_classes = 10
        self.batch_size = 128
        self.n_epochs = 100
        self.save_file = './tmp/'
        if not os.path.exists(self.save_file):
            os.makedirs(self.save_file)
        self.saver = tf.train.Saver()

        # Remove previous Tensors and Operations
        tf.reset_default_graph()

    def model_initialize(self):
        # Import MNIST data
        mnist = input_data.read_data_sets(os.path.join(self.save_file, 'ckpt'),
                                          one_hot=True)

        # Features and Labels
        features = tf.placeholder(tf.float32, [None, n_input])
        labels = tf.placeholder(tf.float32, [None, n_classes])

        # Weights & bias
        weights = tf.Variable(tf.random_normal([n_input, n_classes]))
        bias = tf.Variable(tf.random_normal([n_classes]))

        # Logits - xW + b
        logits = tf.add(tf.matmul(features, weights), bias)

        # Define loss and optimizer
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
            .minimize(cost)

        # Calculate accuracy
        correct_prediction = tf.equal(
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def model_train_and_save(self):
        # Training initialization:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(n_epochs):
                total_batch = math.ceil(mnist.train.num_examples / batch_size)

                # Loop over all batches
                for _ in range(total_batch):
                    batch_features, batch_labels = mnist.train.next_batch(
                        batch_size)
                    sess.run(
                        optimizer,
                        feed_dict={
                            features: batch_features,
                            labels: batch_labels
                        })

                # Print status for every 10 epochs
                if epoch % 10 == 0:
                    valid_accuracy = sess.run(
                        accuracy,
                        feed_dict={
                            features: mnist.validation.images,
                            labels: mnist.validation.labels
                        })
                    print('Epoch {:<3} - Validation Accuracy: {}'.format(
                        epoch, valid_accuracy))

            # Save the model in session
            saver.save(sess, self.save_file)
            print('Trained Model Saved.')

    def model_restore(self):
        saver = tf.train.Saver()
        # Launch the graph
        with tf.Session() as sess:
            saver.restore(sess, save_file)

            test_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features: mnist.test.images,
                    labels: mnist.test.labels
                })

        print('Test Accuracy: {}'.format(test_accuracy))
