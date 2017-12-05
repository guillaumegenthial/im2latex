import collections
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell


from beam_search_decoder_cell import gather_helper


class BSOInput(collections.namedtuple("BSOInput",
    ("ids", "embeddings"))):
    pass


class BSOState(collections.namedtuple("BSOState",
    ("restart", "beam_state", "embeddings", "finished"))):
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


def bso_cross_entropy(logits, pred_ids=None, gold_ids=None, reduce_mode="sum"):
    """
    Args:
        logits: shape = [batch, beam, vocab]
        pred_ids: shape = [batch, beam]
        gold_ids: shape = [batch]

    Returns:
        losses: shape = [batch]

    """
    beam_size = logits.shape[1]
    gold_ids = tf.tile(tf.expand_dims(gold_ids, axis=1), [1, beam_size])
    ce_beams = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gold_ids)
    if reduce_mode == "max":
        losses = tf.reduce_max(ce_beams, axis=-1)
    elif reduce_mode == "sum":
        losses = tf.reduce_sum(ce_beams, axis=-1)
    else:
        raise NotImplementedError("Unknown reduce mode {}".format(reduce_mode))
    return losses


def get_bso_margin(batch_size, beam_size):
    def bso_margin(logits, pred_ids=None, gold_ids=None):
        # shape = [batch, beam]
        pred_logits = gather_helper(logits, pred_ids, batch_size, beam_size, flat=True)
        worst_logit = pred_logits[:, -1]

    return bso_margin


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
        restart = tf.ones(shape=[self._bs_cell._batch_size], dtype=tf.bool)
        beam_state, embeddings, finished = self._bs_cell.initialize()

        return BSOState(restart, beam_state, embeddings, finished)


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
        beam_output, beam_state, embeddings, finished = self._bs_cell.step(state.restart,
                state.beam_state, state.embeddings, state.finished)

        # 2. record violation in the loss
        losses = self._mistake_function(beam_output.logits, pred_ids=beam_output.ids,
                                        gold_ids=inputs.ids)

        # 3. replace embeddings for next time step
        # shape = [batch, beam, embeddings]
        gold_embeddings = tf.tile(tf.expand_dims(inputs.embeddings, axis=1),
                         [1, self._bs_cell._beam_size, 1])
        # shape = [batch, beam]
        gold_ids = tf.tile(tf.expand_dims(inputs.ids, axis=1), [1, self._bs_cell._beam_size])
        # shape = [batch]
        gold_in_beam = tf.reduce_any(tf.equal(gold_ids, beam_output.ids), axis=-1)
        # shape = [batch, beam, embeddings]
        embeddings = tf.where(gold_in_beam, embeddings, gold_embeddings)

        new_output = BSOOutput(losses, beam_output.logits, beam_output.ids, beam_output.parents)
        new_state  = BSOState(gold_in_beam, beam_state, embeddings, finished)

        return (new_output, new_state)


    def __call__(self, inputs, state):
        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)


