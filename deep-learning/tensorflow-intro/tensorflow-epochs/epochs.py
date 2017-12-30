import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
from helper import batches

def print_epoch_stats(epoch_i, sess, last_features, last_labels):
    current_cost = sess.run(cost, feed_dict={features: last_features, labels: last_labels})
    valid_accuracy = sess.run(accuracy, feed_dict={features: valid_features, labels: valid_labels})
    print('Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}'.format(
        epoch_i,
        current_cost,
        valid_accuracy))

n_input = 784
n_classes = 10
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
train_features = mnist.train.images
valid_features = mnist.validation.images
test_features = mnist.test.images
train_labels = mnist.train.labels.astype(np.float32)
valid_labels = mnist.validation.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))
logits = tf.add(tf.matmul(features,weights), bias)
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
batch_size = 128
epochs = 100
learn_rate = 0.001
train_batches = batches(batch_size, train_features, train_labels)
with tf.Session() as sess:
    sess.run(init)
    for epoch_i in range(epochs):
        for batch_features, batch_labels in train_batches:
            train_feed_dict = {
                    features: batch_features,
                    labels: batch_labels,
                    learning_rate: learn_rate }
            sess.run(optimizer, feed_dict=train_feed_dict)
        print_epoch_stats(epoch_i, sess, batch_features, batch_labels)
        test_accuracy = sess.run(
                accuracy,
                feed_dict={features: test_features, labels: test_labels})
print('Test Accuracy: {}'.format(test_accuracy))
