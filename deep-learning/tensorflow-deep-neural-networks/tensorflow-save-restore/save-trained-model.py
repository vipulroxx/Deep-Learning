import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.reset_default_graph()

learning_rate = 0.001
n_input = 784
n_classes = 10

mnist = input_data.read_data_sets('.', one_hot=True)

features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

logits = tf.add(tf.matmul(features, weights), bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

import math

batch_size = 128
n_epochs = 100

save_file = './save_trained_model.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples / batch_size)
        for i in range(int(total_batch)):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict = \
                    {
                        features: batch_features,
                        labels: batch_labels
                    })
        if epoch % 10 == 0:
            valid_accuracy = sess.run(accuracy, feed_dict = \
                    {
                        features: mnist.validation.images,
                        labels: mnist.validation.labels
                    })
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))
    
    saver.save(sess, save_file)
    print('Train Model Saved')

