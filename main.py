import tensorflow as tf
import numpy as np 
from model import *
from util import *
import argparse
import os.path
from generator import *

def train(conf, data, learning_rate=0.001):
	X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
	# h = tf.placeholder(tf.float32, shape=[None, conf.num_classes])
	# if conf.conditional:
		# model = PixelCnn(X, conf, conditional=h)
	# else:
		# model = PixelCnn(X, conf)
	model = PixelCnn(X, conf)

	trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	grads_and_vars = trainer.compute_gradients(model.loss)
	# clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in grads_and_vars]
	# clipped_grad = [(tf.clip_by_value(grad, -conf.grad_clip, conf.grad_clip), var) for grad, var in grads_and_vars]
	# clipped_grad = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in grads_and_vars]
	clipped_grad = grads_and_vars
	optimizer = trainer.apply_gradients(clipped_grad)

	saver = tf.train.Saver(max_to_keep=5)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=(conf.gpu_fraction/100))
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(tf.global_variables_initializer())
		if os.path.exists(conf.model_path + '.meta'):
			saver.restore(sess, conf.model_path)
			print("Model restored from " + conf.model_path)

		for i in range(conf.epochs):
			for j in range(conf.num_batchs):
				if conf.data == 'mnist':
					batch_X, batch_y = data.next_batch(conf.batch_size)
				elif conf.data == 'tag-genome':
					permutation = np.random.permutation(data['labels'].shape[0])
					features = data['features'][permutation]
					labels = data['labels'][permutation]
					batch_X = features[j * conf.batch_size : (j+1) * conf.batch_size]
					batch_y = labels[j * conf.batch_size : (j+1) * conf.batch_size]
				# batch_X = batch_X.reshape([conf.batch_size, conf.img_height, conf.img_width, conf.channel])
				batch_X = (batch_X.reshape([conf.batch_size, conf.img_height, conf.img_width, conf.channel]))
				batch_y = one_hot(batch_y, conf.num_classes)

				if j % 100 == 0:
					print(i, j)
				data_dict = {X : batch_X}
				if conf.conditional:
					data_dict[model.h] = batch_y

				_, cost = sess.run([optimizer, model.loss], feed_dict=data_dict)
			if (i%1 == 0) and (j == conf.num_batchs-1):
				print('Running epoch ' + str(i) + '. Loss: ' + str(cost))
			if (i%1 == 0):
				saver.save(sess, conf.model_path)
				print("Model saved to " + conf.model_path)
			# generate_samples_with_sess(sess, X, model.h, model.pred, conf, 10, 10, 'iter ' + str(i))
		saver.save(sess, conf.model_path)
		print("Model saved to " + conf.model_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, default='tag-genome')
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

	if conf.data == 'mnist':
		from tensorflow.examples.tutorials.mnist import input_data
		data = input_data.read_data_sets(conf.data_path)
		conf.num_classes = 10
		conf.img_height = conf.img_width = 28
		conf.channel = 1
		conf.num_batchs = data.train.num_examples // conf.batch_size
		conf.conditional = True
		train(conf, data.train)
	
	if conf.data == 'tag-genome':
		filename = "tag_relevance.dat"
		num_movies = 9734
		num_tags = 1128
		data = np.zeros(num_movies * num_tags)
		with open(filename, 'r') as infile:
			for i in range(num_movies * num_tags):
				line = infile.readline()
				token = line.split()
				data[i] = float(token[2])
		features = np.zeros((num_movies, num_tags))
		labels = np.zeros((num_movies, 1), dtype=int)

		for i in range(num_movies):
			labels[i] = i
			features[i] = data[i * num_tags : (i+1) * num_tags]
		
		# features_placeholder = tf.placeholder(features.dtype, features.shape)
		# labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
		# dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
		# iterator = dataset.make_initializable_iterator()
		dataset = {'features': features, 'labels': labels}
		# dataset.features = features
		# dataset.labels = labels

		conf.num_classes = num_movies
		conf.img_height = 47
		conf.img_width = 24
		conf.channel = 1
		conf.num_batchs = num_movies // conf.batch_size
		conf.conditional = True
		train(conf, dataset, learning_rate=0.1)