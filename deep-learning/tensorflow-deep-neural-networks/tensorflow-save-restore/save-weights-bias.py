import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from pprint import pprint

save_file = './model.ckpt' # checkpoint

weights = tf.Variable(tf.truncated_normal([2,3]))
bias = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Weights:')
    pprint(sess.run(weights))
    print('Bias:')
    pprint(sess.run(bias))
    saver.save(sess, save_file) #  tf.train.Saver.save()
