import tensorflow as tf 
import numpy as np 
from util import *

def initiate_weights(shape, name):
    initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape, tf.float32, initializer)
    return W

def initiate_bias(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer)

class FullNN():
	#layers is an array of the number of neurons for each layers (exclude input layer and include output layer)
	def __init__(self, X, layers):
		W_arr = []
		b_arr = []
		x_arr = []

		for i in range(len(layers)):
			if i == 0:
				W_arr.append(initiate_weights([X.get_shape()[-1], layers[i]], 'W' + str(i+1)))
				b_arr.append(initiate_bias([layers[i]], 'b' + str(i+1)))
				x_arr.append(tf.nn.relu(tf.matmul(X, W_arr[i]) + b_arr[i]))
			else:
				W_arr.append(initiate_weights([layers[i-1], layers[i]], 'W' + str(i)))
				b_arr.append(initiate_bias([layers[i]], 'b' + str(i)))
				x_arr.append(tf.nn.relu(tf.matmul(x_arr[i-1], W_arr[i]) + b_arr[i]))
		self.output = x_arr[-1]

#not tested
class ConvEncoder():
	def __init__(self, X, layers, img_height, img_width, num_classes, num_channel=1):
		W_arr = []
		b_arr = []
		conv_arr = []
		pool_arr = []
		
		for i in range(len(layers)):
			W_arr.append(initiate_weights(layers[i], 'W_conv' + str(i)))
			b_arr.append(initiate_bias([layers[-1]], 'b_conv' + str(i)))
			if i == 0:
				conv_arr.append(tf.nn.relu(conv2d(X, W_arr[i]) + b_arr[i]))
			else:
				conv_arr.append(tf.nn.relu(conv2d(pool_arr[i-1], W_arr[i]) + b_arr[i]))
			if i < len(layers)-1:
				pool_arr.append(max_pool_2x2(conv_arr[i]))
				img_height /= 2
				img_width /= 2

		conv_flat = tf.reshape(conv_arr[len(layers)-1], (-1, img_height*img_width*layers[-1][-1]))
		W_fc = initiate_weights([img_height*img_width*layers[-1][-1], num_classes], 'W_fc')
		b_fc = initiate_bias([num_classes], 'b_fc')
		self.pred = tf.nn.softmax(tf.matmul(conv_flat, W_fc) + b_fc)
