import tensorflow as tf 
import numpy as np 

def one_hot(batch, num_classes):
	ret = np.zeros((batch.shape[0], num_classes))
	ret[np.arange(batch.shape[0]), batch] = 1
	return ret