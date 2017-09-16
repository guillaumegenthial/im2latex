import numpy as np
import tensorflow as tf
from utils.tf import conv2d, max_pooling2d, batch_normalization
from tensorflow.contrib.rnn import GRUCell, LSTMCell

class Encoder(object):
    def __init__(self, config):
        self.config = config


    def __call__(self, training, img, dropout):
        """
        Args:
            training: (tf.placeholder) tf.bool
            img: batch of img, shape = (?, height, width, channels)
        Returns:
            the encoded images, shape = (?, h', w', c')
        """        
        img = tf.cast(img, tf.float32) / 255.

        """
        ISSUE: seems to have performance issue (speed and accuracy) when
        using batch norm 
        TODO: check that the training bool is what we want
        """

        with tf.variable_scope("convolutional_encoder"):
            out = conv2d(inputs=img, filters=64, kernel_size=3) 
            out = max_pooling2d(inputs=out)

            out = conv2d(inputs=out, filters=128, kernel_size=3)
            out = max_pooling2d(inputs=out)

            out = conv2d(inputs=out, filters=256, kernel_size=3)

            out = conv2d(inputs=out, filters=256, kernel_size=3)

            if self.config.encode_mode == "vanilla":
                out = max_pooling2d(inputs=out, pool_size=(2,1), strides=(2,1)) 

            out = conv2d(inputs=out, filters=512, kernel_size=3)

            if self.config.encode_mode == "vanilla":
                out = max_pooling2d(inputs=out, pool_size=(1,2), strides=(1,2))

            if self.config.encode_mode == "cnn":
                out = conv2d(inputs=out, filters=512, kernel_size=(2, 4), strides=(2, 2))

            out = conv2d(inputs=out, filters=512, kernel_size=3, padding='VALID')

        if self.config.encode_with_lstm:
            with tf.variable_scope("bilstm_encoder"):
                N = tf.shape(out)[0]
                H_out = tf.shape(out)[1]
                W_out = tf.shape(out)[2]
                C_out = out.shape[-1].value

                inputs = tf.reshape(out, shape=(N*H_out, W_out, C_out))

                cell_fw = LSTMCell(self.config.encoder_dim)
                cell_bw = LSTMCell(self.config.encoder_dim)

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32)
                outputs_fw, outputs_bw = outputs

                out = tf.concat([outputs_fw, outputs_bw], axis=2)
                out = tf.reshape(out, shape=(N, H_out, W_out, 2*self.config.encoder_dim))

        return out
