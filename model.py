import tensorflow as tf
import numpy as py

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def initiate_weights(shape, name, mask_type=None):
    initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape, tf.float32, initializer)

    if mask_type is not None: 
        x_mid = shape[0]//2 
        y_mid = shape[0]//2
        mask = np.one(shape, dtype=np.float32)
        mask[x_mid, y_mid+1:, :, :] = 0
        mask[x_mid+1:, :, :, :] = 0
        W *= mask

    return W

def initiate_unconditional_bias(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer)

def initiate_conditional_bias(conditional, W_shape, name=''):
    h_shape = conditional.get_shape()
    V = initiate_weights([(int)(h_shape[1]), W_shape[3]], name)
    b = tf.matmul(conditional, V)
    b_shape = tf.shape(b)
    return tf.reshape(b, (b_shape[0], 1, 1, b_shape[1]))

class PixelCnn():
    def __init__(self, filter_shape, input, out_dim, gated=True, mask_type=None, conditional=None):
        self.input = input
        self.W_shape = [filter_shape[0], filter_shape[0], input.get_shape[1], out_dim]
        self.b_shape = out_dim
        self.gated = gated
        self.conditional = conditional
        self.mask_type = mask_type

    def gated_activation_unit(self):
        W_f = initiate_weights(self.W_shape, "W_f", mask_type = self.mask_type)
        W_g = initiate_weights(self.W_shape, "W_g", mask_type = self.mask_type)
        if self.conditional is not None:
            b_f = initiate_conditional_bias(self.conditional, self.W_shape, 'V_f')
            b_g = initiate_conditional_bias(self.conditional, self.W_shape, 'V_g')
        else :
            b_f = initiate_unconditional_bias(self.b_shape, 'b_f')
            b_g = initiate_unconditional_bias(self.b_shape, 'b_g')

        tanh_factor = tf.tanh(conv2d(self.input, W_f))
        sigmoid_factor = tf.sigmoid(conv2d(self.input, W_g) + b_g)
        self.output = tf.multiply(tanh_factor, sigmoid_factor)

    def get_output(self):
        return self.output


