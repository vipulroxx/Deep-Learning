import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def run():
    output = None
    logit_data = [2.0, 1.0, 0.1, 0.7, -2.0]
    logits = tf.placeholder(tf.float32)
    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        output = sess.run(softmax, feed_dict={logits: logit_data})
    print(output)

run()
