import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from pprint import pprint

save_file = './model.ckpt'

tf.reset_default_graph()

# restore() loads the saved data to weights and bias
# therfore create weights and bias again
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

with tf.Session() as sess:

    # tf.train.Saver.restore() sets all the TensorFlow Variables
    # therefore no need to call tf.global_variables_initializer().

    saver.restore(sess, save_file)
    print('Weight:')
    pprint(sess.run(weights))
    print('Bias:')
    pprint(sess.run(bias))
