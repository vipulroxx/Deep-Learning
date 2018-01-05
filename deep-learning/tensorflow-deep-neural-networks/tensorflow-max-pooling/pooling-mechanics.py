conv_layer = tf.nn.conv2d(input, weight, strides=[1,2,2,1], padding = 'SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
# Apply max pooling
conv_layer = tf.nn.max_pool(\
        conv_layer,
        # batch and size are typically set to 1 for both ksize and strides
        ksize = [1,2,2,1], # ksize is the size of the filter 2 x 2
        strides = [1,2,2,1], #length of stride 2 x 2
        padding = 'SAME')

# Example
input = tf.placeholder(tf.float32, (None, 4, 4, 5))
filter_shape = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
padding = 'VALID'
pool = tf.nn.max_pool(input, filter_shape, strides, padding)
