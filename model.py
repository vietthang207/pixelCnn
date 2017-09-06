import tensorflow as tf
import numpy as py

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def initiate_masked_weights(shape, name):
    initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape, tf.float32, initializer)

    x_mid = shape[0]//2
    y_mid = shape[0]//2
    mask = np.one(shape, dtype=np.float32)
    mask[x_mid, y_mid+1:, :, :] = 0
    mask[x_mid+1:, :, :, :] = 0
    W *= mask

    return W

def initiate_bias(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer)

class PixelCnn():
    def __init__(self, filter_shape, input, out_dim, gated=True, conditional=None):
        self.input = input
        self.W_shape = [filter_shape[0], filter_shape[0], input.get_shape[1], out_dim]
        self.b_shape = out_dim
        self.gated = gated
        self.conditional = conditional

    def gated_activation_unit(self):
    