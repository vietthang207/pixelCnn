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
        if mask_type == 'a':
            mask[x_mid, y_mid, :, :] = 0        
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

class GatedConvLayer():
    # payload is for stacking up
    def __init__(self, filter_shape, input, out_dim, gated=True, mask_type=None, conditional=None, payload=None, is_activated=True):
        self.input = input
        self.W_shape = [filter_shape[0], filter_shape[1], input.get_shape[1], out_dim]
        self.b_shape = out_dim
        self.gated = gated
        self.conditional = conditional
        self.mask_type = mask_type
        self.payload = payload
        self.is_activated = is_activated

        if gated:
            gated_activation_unit()
        else:
            non_gated_activation_unit()

    def gated_activation_unit(self):
        W_f = initiate_weights(self.W_shape, "W_f", mask_type = self.mask_type)
        W_g = initiate_weights(self.W_shape, "W_g", mask_type = self.mask_type)
        if self.conditional is not None:
            b_f = initiate_conditional_bias(self.conditional, self.W_shape, 'V_f')
            b_g = initiate_conditional_bias(self.conditional, self.W_shape, 'V_g')
        else :
            b_f = initiate_unconditional_bias(self.b_shape, 'b_f')
            b_g = initiate_unconditional_bias(self.b_shape, 'b_g')

        f_term = conv2d(self.input, W_f) + b_f
        g_term = conv2d(self.input, W_g) + b_g
        if self.payload is not None:
            f_term += self.payload
            g_term += self.payload

        self.output = tf.multiply(tf.tanh(f_term), tf.sigmoid(g_term))

    def non_gated_activation_unit(self):
        W = initiate_weights(self.W_shape, 'W', mask_type = self.mask_type)
        b = initiate_unconditional_bias(self.b_shape, 'b')
        self.output = tf.add(conv2d(self.input, W), b)
        if self.is_activated:
            self.output = tf.nn.relu(self.output)

    def get_output(self):
        return self.output

class PixelCnn():
    def __init__(self, X, conf, conditional=None):
        self.X = X
        self.conf = conf
        self.h = conditional

        v_stack_in = X
        h_stack_in = X
        f_map = conf.f_map
        for i in range(conf.num_layers):
            filter_size = 3 if i>0 else 7
            mask_type = 'b' if i>0 else 'a'
            residual = (i>0)

            with tf.variable_scope('v_stack_in' + str(i)):
                v_stack_in = GatedConvLayer([filter_size, filter_size], v_stack_in, f_map, mask_type=mask_type, conditional=self.h).get_output()
            with tf.variable_scope('v_stack_out' + str(i)):
                v_stack_out = GatedConvLayer([1, 1], v_stack_in, f_map, mask_type=mask_type, gated=False).get_output()

            with tf.variable_scope('h_stack' + str(i)):
                h_stack = GatedConvLayer([1, filter_size], h_stack_in, f_map, payload=v_stack_out, mask_type=mask_type, conditional=self.h).get_output()
            with tf.variable_scope('h_stack_out' + str(i)):
                h_stack_out = GatedConvLayer([1, 1], h_stack, f_map, gated=False, mask_type=mask_type).get_output()
                if residual:
                    h_stack_out += h_stack_in

        with tf.variable_scope('fc_1'):
            fc1 = GatedConvLayer([1, 1], h_stack_in, f_map, gated=False, mask_type='b').get_output()

        with tf.variable_scope('fc_2'):
            fc2 = GatedConvLayer([1, 1], fc1, 1, gated=False, is_activated=False, mask_type='b').get_output()
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2, labels=self.X))
        self.pred = tf.nn.sigmoid(self.fc2)
        
