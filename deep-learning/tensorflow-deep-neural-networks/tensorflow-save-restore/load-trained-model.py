import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

save_file = './save_trained_model.ckpt'
tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

n_input = 784  
n_classes = 10

mnist = input_data.read_data_sets('.', one_hot=True)
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

logits = tf.add(tf.matmul(features, weights), bias)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, save_file)
    test_accuracy = sess.run(accuracy, feed_dict= \
            {
                features: mnist.test.images,
                labels: mnist.test.labels
            })
print('Test Accuracy: {}'.format(test_accuracy))
