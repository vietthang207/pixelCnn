import tensorflow as tf
import numpy as np 
from model import *
from util import *
import argparse

def train(conf, data, learning_rate=0.001):
	X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
	h = tf.placeholder(tf.float32, shape=[None, conf.num_classes])
	if conf.conditional:
		model = PixelCnn(X, conf, conditional=h)
	else:
		model = PixelCnn(X, conf)

	trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	grads_and_vars = trainer.compute_gradients(model.loss)
	#clipped_grad = [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.grad_clip, 'c'), gv[1]) for gv in grads_and_vars]
	clipped_grad = grads_and_vars
	optimizer = trainer.apply_gradients(clipped_grad)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(conf.epochs):
			for j in range(conf.num_batchs):
				batch_X, batch_y = data.train.next_batch(conf.batch_size)
				batch_X = batch_X.reshape([conf.batch_size, conf.img_height, conf.img_width, conf.channel])
				batch_y = one_hot(batch_y, conf.num_classes)

				data_dict = {X : batch_X}
				if conf.conditional:
					data_dict[h] = batch_y

				_, cost = sess.run([optimizer, model.loss], feed_dict=data_dict)
				if (i%1 == 0) and (j == conf.num_batchs-1):
					print('Running epoch ' + str(i) + '. Loss: ' + str(cost))


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
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--samples_path', type=str, default='samples')
    parser.add_argument('--summary_path', type=str, default='logs')
    conf = parser.parse_args()

    from tensorflow.examples.tutorials.mnist import input_data
    data = input_data.read_data_sets(conf.data_path)
    conf.num_classes = 10
    conf.img_height = conf.img_width = 28
    conf.channel = 1
    conf.num_batchs = data.train.num_examples // conf.batch_size

    conf.conditional = True
    train(conf, data)