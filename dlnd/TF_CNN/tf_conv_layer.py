#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def tf_conv_layer():
    # Output depth
    k_output = 64

    # Image Properties
    image_width = 10
    image_height = 10
    color_channels = 3

    # Convolution filter
    filter_size_width = 5
    filter_size_height = 5

    # Input/Image
    input = tf.placeholder(
        tf.float32, shape=[None, image_height, image_width, color_channels])

    # Weight and bias
    weight = tf.Variable(
        tf.truncated_normal(
            [filter_size_height, filter_size_width, color_channels, k_output]))
    bias = tf.Variable(tf.zeros(k_output))

    # Apply Convolution
    # conv_layer original size: (?, 5, 5, 64)
    conv_layer = tf.nn.conv2d(
        input, weight, strides=[1, 2, 2, 1], padding='SAME')
    # Add bias
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    # Apply activation function
    conv_layer = tf.nn.relu(conv_layer)
    print(conv_layer.shape)

    # conv_layer pooled size: (?, 3, 3, 64)
    conv_layer = tf.nn.max_pool(
        conv_layer,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')

    print(conv_layer.shape)


if __name__ == '__main__':
    tf_conv_layer()
