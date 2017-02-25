# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10.0)
y = tf.constant(2.0)
# Also works:
z = x/y - 1
# z = tf.subtract(tf.divide(x, y), tf.constant(1.0))

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)
