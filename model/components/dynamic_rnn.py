import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn


def transpose_batch_time(t):
    if t.shape.ndims == 2:
        return tf.transpose(t, [1, 0])
    elif t.shape.ndims == 3:
        return tf.transpose(t, [1, 0, 2])
    elif t.shape.ndims == 4:
        return tf.transpose(t, [1, 0, 2, 3])
    else:
        raise NotImplementedError


def dynamic_rnn(decoder_cell, inputs, initial_state):
    """
    Args:
        decoder_cell: (instance of DecoderCell) with step method
                step(inputs, state)
        inputs: (possibly nested structure) of tensors of
                shape = [batch, max_time_steps, ...]
        initial_state: (tensor) instance of decodercell state

    """
    # create TA for outputs by mimicing the structure of decodercell output
    def create_ta(d):
        return tf.TensorArray(dtype=d, size=0, dynamic_size=True)

    initial_time = tf.constant(0, dtype=tf.int32)
    try:
        n_steps = tf.shape(inputs[0])[1] # nested case
    except:
        n_steps = tf.shape(inputs)[1] # non nested case

    initial_outputs_ta = nest.map_structure(create_ta, decoder_cell.output_dtype)

    def condition(time, unused_ta, unused_state):
        return time < n_steps

    def body(time, outputs_ta, state):
        inputs_t = nest.map_structure(lambda t: t[:, time], inputs)
        new_output, new_state = decoder_cell.step(inputs_t, state)

        outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                      outputs_ta, new_output)

        return (time + 1, outputs_ta, new_state)

    with tf.variable_scope("rnn"):
        res = tf.while_loop(
            condition,
            body,
            loop_vars=[initial_time, initial_outputs_ta, initial_state])

    # get final outputs and states
    final_outputs_ta, final_state = res[1], res[2]

    # unfold and stack the structure from the nested tas
    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

    # transpose the final output
    final_outputs = nest.map_structure(transpose_batch_time, final_outputs)

    return final_outputs, final_state