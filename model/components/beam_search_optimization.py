import collections
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


class BSOState(collections.namedtuple("BSOState",
    ("time", "beam_state", "embedding", "finished"))):
    pass


class BSOCell(RNNCell):

    def __init__(self, beam_search_cell):
        self._bs_cell = beam_search_cell
        self._state_size = BSOState(tf.TensorShape(1), self._bs_cell.state_size,
                self._bs_cell.inputs_size, self._bs_cell.finished_size)
        self._output_size = self._bs_cell.output_size


    @property
    def state_size(self):
        return self._state_size


    @property
    def output_size(self):
        return self._output_size


    @property
    def output_dtype(self):
        return self._bs_cell.output_dtype


    def initial_state(self):
        time = tf.constant(0, tf.int32)
        beam_state, embedding, finished = self._bs_cell.initialize()

        return BSOState(time, beam_state, embedding, finished)


    def step(self, inputs, state):
        time, beam_state, embedding, finished = state

        gold_embedding = tf.tile(tf.expand_dims(inputs, axis=1),
                         [1, self._bs_cell._beam_size, 1])

        # 1. if gold hypothesis fell out of the beam, add loss + embeddings

        # 2. if violation, record loss for the time step

        output, beam_state, embedding, finished = self._bs_cell.step(time,
                beam_state, gold_embedding, finished)

        logits, ids, parents = output

        new_state = BSOState(time+1, beam_state, embedding, finished)

        return (output, new_state)


    def __call__(self, inputs, state):
        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)


