import collections
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell


class BSOInput(collections.namedtuple("BSOInput",
    ("id", "embedding"))):
    pass


class BSOState(collections.namedtuple("BSOState",
    ("time", "beam_state", "embedding", "finished"))):
    pass


class BSOOutput(collections.namedtuple("BSOOutput",
    ("losses", "logits", "ids", "parents"))):
    pass


def get_inputs(ids, embeddings):
    """
    Args:
        ids: shape = [batch_size]
        embeddings: shape = [batch_size, embedding_size]

    """
    return nest.map_structure(lambda i, e: BSOInput(i, e), ids, embeddings)


def bso_cross_entropy(logits, ids, reduce_mode="sum"):
    """
    Args:
        logits: shape = [batch, beam, vocab]
        ids: shape = [batch]

    Returns:
        losses: shape = [batch]

    """
    beam_size = logits.shape[1]
    ids = tf.tile(tf.expand_dims(ids, axis=1), [1, beam_size])
    ce_beams = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ids)
    if reduce_mode == "max":
        losses = tf.reduce_max(ce_beams, axis=-1)
    elif reduce_mode == "sum":
        losses = tf.reduce_sum(ce_beams, axis=-1)
    else:
        raise NotImplementedError("Unknown reduce mode {}".format(reduce_mode))
    return losses


class BSOCell(RNNCell):

    def __init__(self, beam_search_cell, mistake_function):
        """
        Args:
            beam_search_cell: instance of BeamSearchCell with step function
            mistake_function: function of
                                    logits: [batch_size, beam_size, vocab_size]
                                    ids: [batch_size]
                              returns:
                                    losses: [batch_size]

        """
        self._bs_cell = beam_search_cell
        self._state_size = BSOState(tf.TensorShape(1), self._bs_cell.state_size,
                self._bs_cell.inputs_size, self._bs_cell.finished_size)
        self._output_size = self._bs_cell.output_size
        self._mistake_function = mistake_function


    @property
    def state_size(self):
        return self._state_size


    @property
    def output_size(self):
        return self._output_size


    @property
    def output_dtype(self):
        bs_cell_dtype = self._bs_cell.output_dtype
        return BSOOutput(tf.float32, bs_cell_dtype.logits, bs_cell_dtype.ids, bs_cell_dtype.parents)


    def initial_state(self):
        time = tf.constant(0, tf.int32)
        beam_state, embedding, finished = self._bs_cell.initialize()

        return BSOState(time, beam_state, embedding, finished)


    def step(self, inputs, state):
        """
        Args:
            inputs: token at time step t, named_tuple of
                    ids: shape = [batch_size]
                    embeddings: shape = [batch_size, embedding_size]
                embeddings of gold words from time step t
            state: instance of BSOState from previous time step t-1

        """
        # 1. compute prediction
        beam_output, beam_state, embedding, finished = self._bs_cell.step(state.time,
                state.beam_state, state.embedding, state.finished)

        # 2. record violation in the loss
        losses = self._mistake_function(beam_output.logits, inputs.id)


        # 3. replace embeddings for next time step
        gold_embedding = tf.tile(tf.expand_dims(inputs.embedding, axis=1),
                         [1, self._bs_cell._beam_size, 1])


        new_output = BSOOutput(losses, beam_output.logits, beam_output.ids, beam_output.parents)
        new_state  = BSOState(state.time+1, beam_state, embedding, finished)

        return (new_output, new_state)


    def __call__(self, inputs, state):
        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)


