#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def tf_conv_padding():
    input = tf.placeholder(tf.float32, (None, 32, 32, 3))
    filter_weights = tf.Variable(tf.truncated_normal(
        (8, 8, 3, 20)))  # (height, width, input_depth, output_depth)
    filter_bias = tf.Variable(tf.zeros(20))
    strides = [1, 2, 2, 1]  # (batch, height, width, depth)
    padding = 'VALID'
    # Valid: (?, 13, 13, 20)
    conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
    return conv


if __name__ == '__main__':
    res = tf_conv_padding()
    print(res.shape)
