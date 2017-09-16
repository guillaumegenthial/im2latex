import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_normal_initializer

def conv2d(inputs, filters=64, kernel_size=3, strides=1, padding='SAME'):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding, 
        activation=tf.nn.relu)


def max_pooling2d(inputs, pool_size=2, strides=2, padding="SAME"):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training, axis=-1)



def weight_initializer():
    def _initializer(shape, dtype, partition_info=None):
        if len(shape) == 2 and (shape[0] == shape[1]):
            return tf.orthogonal_initializer()(shape, dtype, partition_info)
        else:
            return glorot_normal_initializer()(shape, dtype, partition_info)

    return _initializer
