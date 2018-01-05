# Just syntax

'''
H = height, W = width, D = depth

We have an input of shape 32x32x3 (HxWxD)
20 filters of shape 8x8x3 (HxWxD)
A stride of 2 for both the height and width (S)
With padding of size 1 (P)
'''
 
input = tf.placeholer(tf.float32, (None, 32,32,3))
filter_weights = tf.Variable(tf.truncated_normal((8,8,3,20))) # (height, width, input_depth, output_depth = number of filters)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1,2,2,1] # (batch, height, width, depth)
padding = 'SAME' # can be 'VALID' as well
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
