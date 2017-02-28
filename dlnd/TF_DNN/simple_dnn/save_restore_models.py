#!/usr/bin/env python

import math
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class ModelSaveRestore(object):
    def __init__(self):
        # Hyper-parameters' setting
        self.learning_rate = 0.001
        self.n_input = 784
        self.n_classes = 10
        self.batch_size = 128
        self.n_epochs = 100
        self.save_file = './tmp/'
        # Import MNIST data
        if not os.path.exists(self.save_file):
            os.makedirs(self.save_file)
        self.data = input_data.read_data_sets(
            os.path.join(self.save_file, 'datasets'), one_hot=True)

        # Remove previous Tensors and Operations
        tf.reset_default_graph()
        print('Static Init Done!')

    def model_initialize(self):
        model_init_flag = False
        # Features and Labels
        self.features = tf.placeholder(tf.float32, [None, self.n_input])
        self.labels = tf.placeholder(tf.float32, [None, self.n_classes])

        # Weights & bias: name for fine-tuning
        self.weights = tf.Variable(
            tf.random_normal([self.n_input, self.n_classes]), name='weights_0')
        self.bias = tf.Variable(
            tf.random_normal([self.n_classes]), name='bias_0')

        # Logits - xW + b
        logits = tf.add(tf.matmul(self.features, self.weights), self.bias)

        # Define loss and optimizer
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels))
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(cost)

        # Calculate accuracy
        correct_prediction = tf.equal(
            tf.argmax(logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        model_init_flag = not model_init_flag
        return model_init_flag

    def model_train_and_save(self):
        train_save_flag = False
        # Training initialization:
        init = tf.global_variables_initializer()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(self.n_epochs):
                total_batch = math.ceil(self.data.train.num_examples /
                                        self.batch_size)

                # Loop over all batches
                for _ in range(total_batch):
                    batch_features, batch_labels = self.data.train.next_batch(
                        self.batch_size)
                    sess.run(
                        self.optimizer,
                        feed_dict={
                            self.features: batch_features,
                            self.labels: batch_labels
                        })

                # Print status for every 10 epochs
                if epoch % 10 == 0:
                    valid_accuracy = sess.run(
                        self.accuracy,
                        feed_dict={
                            self.features: self.data.validation.images,
                            self.labels: self.data.validation.labels
                        })
                    print('Epoch {:<3} - Validation Accuracy: {}'.format(
                        epoch, valid_accuracy))

            # Save the model in session
            tf.train.Saver().save(sess, self.save_file)
            print('Trained Model Saved.')

        train_save_flag = not train_save_flag
        return train_save_flag

    def model_restore(self):
        restore_flag = False
        # Launch the graph
        with tf.Session() as sess:
            tf.train.Saver().restore(sess, self.save_file)

            test_accuracy = sess.run(
                self.accuracy,
                feed_dict={
                    self.features: self.data.test.images,
                    self.labels: self.data.test.labels
                })

        print('Test Accuracy: {}'.format(test_accuracy))

        restore_flag = not restore_flag
        return restore_flag

    @staticmethod
    def run():
        case = ModelSaveRestore()
        model_init = case.model_initialize()
        assert model_init
        model_train_save = case.model_train_and_save()
        assert model_train_save
        model_restore = case.model_restore()
        assert model_restore


if __name__ == '__main__':
    ModelSaveRestore.run()
