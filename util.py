import tensorflow as tf 
import numpy as np 

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
def one_hot(batch, num_classes):
	ret = np.zeros((batch.shape[0], num_classes))
	ret[np.arange(batch.shape[0]), batch] = 1
	return ret

def binarize(images):
	return (np.random.uniform(size=images.shape) < images).astype(np.float32)