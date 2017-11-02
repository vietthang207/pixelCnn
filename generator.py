import tensorflow as tf
import numpy as np 
from model import *
from util import *
import argparse
import os.path
from datetime import datetime
import scipy.misc

def save_images(images, n_row, n_col, conf, description):
	images = images.reshape((n_row, n_col, conf.img_height, conf.img_width))
	images = images.transpose(1, 2, 0, 3)
	images = images.reshape((conf.img_height * n_row, conf.img_width * n_col))

	filename = datetime.now().strftime('%Y_%m_%d_%H_%M')+description+".jpg"
	scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(os.path.join(conf.samples_path, filename))

def save_movies(samples, tags, conf, description):
	samples = samples.reshape(-1, conf.img_height * conf.img_width * conf.channel)
	filename = datetime.now().strftime('%Y_%m_%d_%H_%M')+description+".txt"
	f = open(os.path.join(conf.samples_path, filename), 'w')
	for i in range(len(samples)):
		for j in range(conf.img_height * conf.img_width * conf.channel):
			# print(len(samples), len(tags))
			f.write(tags[j][-1])
			f.write(': ' + str(samples[i][j]) + '\n')
	f.close()

def generate_samples(conf, n_row, n_col, description):
	X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
	h = tf.placeholder(tf.float32, shape=[None, conf.num_classes])
	if conf.conditional:
		model = PixelCnn(X, conf, conditional=h)
	else:
		model = PixelCnn(X, conf)

	samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
	labels = one_hot(np.array([0,1,2,3,4,5,6,7,8,9]*10), conf.num_classes)
	saver = tf.train.Saver(max_to_keep=5)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=(conf.gpu_fraction/100))
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		if os.path.exists(conf.model_path + '.meta'):
			saver.restore(sess, conf.model_path)
			print("Model restored from " + conf.model_path)
		else:
			print("No model at " + conf.model_path)

		for i in range(conf.img_height):
			for j in range(conf.img_width):
				for k in range(conf.channel):
					data_dict = {X:samples}
					next_sample = sess.run(model.pred, feed_dict=data_dict)
					next_sample = binarize(next_sample)
					samples[:, i, j, k] = next_sample[:, i, j, k]
		save_images(samples, n_row, n_col, conf, description)

def generate_samples_with_sess(sess, X, h, pred, conf, n_row, n_col, description, tags=None):
	print('Generating with description: ', description)
	samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
	print('sample shape', samples.shape)
	labels = one_hot(np.array([0,1,2,3,4,5,6,7,8,9]*10), conf.num_classes)

	for i in range(conf.img_height):
		print(i)
		for j in range(conf.img_width):
			for k in range(conf.channel):
				data_dict = {X:samples}
				if conf.conditional is True:
					data_dict[h] = labels
				next_sample = sess.run(pred, feed_dict=data_dict)
				# next_sample = binarize(next_sample)
				samples[:, i, j, k] = next_sample[:, i, j, k]
	if conf.data == 'mnist':
		save_images(samples, n_row, n_col, conf, description)
	elif conf.data == 'tag-genome':
		save_movies(samples, tags, conf, description)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--f_map', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--grad_clip', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--model_path', type=str, default='./savedModel/model.ckpt')
    parser.add_argument('--samples_path', type=str, default='./generatedSample')
    parser.add_argument('--gpu_fraction', type=int, default=100)
    parser.add_argument('--summary_path', type=str, default='logs')
    conf = parser.parse_args()

    conf.num_classes = 10
    conf.img_height = conf.img_width = 28
    conf.channel = 1
    conf.conditional = True
    generate_samples(conf, 10, 10, 'bla')