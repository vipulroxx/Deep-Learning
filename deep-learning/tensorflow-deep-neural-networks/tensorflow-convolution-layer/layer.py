import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Output depth
k_output = 64

# Image properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(tf.float32, shape = [None, \
        image_height, image_width, color_channels])

# Weight and Bias
weight = tf.Variable(tf.truncated_normal(\
        [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, \
        strides = [1,2,2,1], padding = 'SAME') # Weight as filter [batch, input_height, input_width, input_channels] batch and input_channels are generally set to 1

# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias) # adds a 1-d bias to the last dimension in a matrix.

# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
