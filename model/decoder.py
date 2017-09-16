import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from .dynamic_decode import dynamic_decode
from .attention_mechanism import AttentionMechanism
from .attention_cell import AttentionCell
from .greedy_decoder_cell import GreedyDecoderCell
from .beam_search_decoder_cell import BeamSearchDecoderCell
from utils.tf import weight_initializer


class Decoder(object):
    """
    Implements this paper https://arxiv.org/pdf/1609.04938.pdf
    """
    def __init__(self, config):
        self.config = config


    def __call__(self, training, encoded_img, formula, dropout):
        """
        Args:
            training: (tf.placeholder) bool
            encoded_img: (tf.Tensor) shape = (N, H, W, C)
            formula: (tf.placeholder), shape = (N, T)
        Returns:
            pred_train: (tf.Tensor), shape = (?, ?, vocab_size) logits of each class
            pret_test: (structure) 
                - pred.test.logits, same as pred_train
                - pred.test.ids, shape = (?, config.max_length_formula)
        """
        # get embeddings for training
        
        if self.config.pretrained_embeddings:
            print("Reloading pretrained embeddings")
            npz_file = np.load(self.config.path_embeddings)
            embeddings = npz_file["arr-0"]
            assert(embeddings.shape == [self.config.vocab_size, self.config.dim_embeddings])
            E = tf.get_variable("E", shape=embeddings.shape,
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(embeddings),
                                  trainable=self.config.trainable_embeddings)
        else:
            E = tf.get_variable("E", shape=[self.config.vocab_size, self.config.dim_embeddings], 
            dtype=tf.float32, initializer=embedding_initializer())

        start_token = tf.get_variable("start_token", shape=[self.config.dim_embeddings],
            dtype=tf.float32, initializer=embedding_initializer())

        # embedding with start token
        batch_size        = tf.shape(formula)[0]
        embedding_formula = tf.nn.embedding_lookup(E, formula)
        start_token_      = tf.reshape(start_token, [1, 1, self.config.dim_embeddings])
        start_tokens      = tf.tile(start_token_, multiples=[batch_size, 1, 1])
        embedding_train   = tf.concat([start_tokens, embedding_formula[:, :-1, :]], axis=1) 


        # attention cell
        with tf.variable_scope("attn_cell", reuse=False, initializer=weight_initializer()):
            attention_mechanism = AttentionMechanism(encoded_img, self.config.attn_cell_config["dim_e"])
            cell      = LSTMCell(self.config.attn_cell_config["num_units"])
            attn_cell = AttentionCell(cell, attention_mechanism, dropout, self.config.attn_cell_config)
            train_outputs, _ = tf.nn.dynamic_rnn(attn_cell, embedding_train, initial_state=attn_cell.initial_state())

        with tf.variable_scope("attn_cell", reuse=True):
            attention_mechanism = AttentionMechanism(encoded_img, self.config.attn_cell_config["dim_e"])
            cell         = LSTMCell(self.config.attn_cell_config["num_units"], reuse=True)
            attn_cell    = AttentionCell(cell, attention_mechanism, dropout, self.config.attn_cell_config)
            decoder_cell = GreedyDecoderCell(E, attn_cell, batch_size, start_token, self.config.id_END)

            test_outputs, _ = dynamic_decode(decoder_cell, self.config.max_length_formula+1)
                
            if self.config.decoding == "beam_search":
                attention_mechanism = AttentionMechanism(
                    img=encoded_img, 
                    dim_e=self.config.attn_cell_config["dim_e"], 
                    tiles=self.config.beam_size)
                
                cell         = LSTMCell(self.config.attn_cell_config["num_units"], reuse=True)
                attn_cell    = AttentionCell(cell, attention_mechanism, dropout, self.config.attn_cell_config)

                decoder_cell = BeamSearchDecoderCell(E, attn_cell, batch_size, 
                        start_token, self.config.beam_size, self.config.id_END)

                beam_search_outputs, _ = dynamic_decode(decoder_cell, self.config.max_length_formula+1)
                
                # concatenate beam search outputs with the greedy outputs
                # greedy outputs comes last
                time_greedy = tf.shape(test_outputs.ids)[1]
                time_beam = tf.shape(beam_search_outputs.ids)[1]

                test_outputs = nest.map_structure(
                    lambda t, d: pad(t, d, time_beam - time_greedy),
                    test_outputs,
                    decoder_cell.final_output_dtype)

                beam_search_outputs = nest.map_structure(
                    lambda t, d: pad(t, d, time_greedy - time_beam),
                    beam_search_outputs,
                    decoder_cell.final_output_dtype)

                test_outputs = nest.map_structure(
                    lambda t1, t2: tf.concat([t1, tf.expand_dims(t2, axis=2)], axis=2),
                    beam_search_outputs, test_outputs)
        
        return train_outputs, test_outputs


def pad(t, d, time_diff):
    def _pad(t, d):
        batch_size = tf.shape(t)[0]

        if t.shape.ndims == 2:
            pad_array = tf.zeros([batch_size, time_diff], dtype=d)
        elif t.shape.ndims == 3:
            pad_array = tf.zeros([batch_size, time_diff, tf.shape(t)[2]], dtype=d)
        elif t.shape.ndims == 4:
            pad_array = tf.zeros([batch_size, time_diff, tf.shape(t)[2], tf.shape(t)[3]], dtype=d)
        else:
            raise NotImplementedError

        return tf.concat([t, pad_array], axis=1)

    return tf.cond(
        time_diff > 0,
        lambda: _pad(t, d),
        lambda: t)


def embedding_initializer():
    def _initializer(shape, dtype, partition_info=None):
        E = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        E = tf.nn.l2_normalize(E, -1)

        return E

    return _initializer