import collections
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell


from beam_search_decoder_cell import gather_helper


class BSOInput(collections.namedtuple("BSOInput",
    ("ids", "embeddings"))):
    pass


class BSOState(collections.namedtuple("BSOState",
    ("restarts", "gold_beam", "beam_state", "embeddings", "finished"))):
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


def bso_cross_entropy(logits, gold_ids, reduce_mode="min", **kwargs):
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
    elif reduce_mode == "min":
        losses = tf.reduce_min(ce_beams, axis=-1)
    else:
        raise NotImplementedError("Unknown reduce mode {}".format(reduce_mode))
    return losses


def bso_loss(margins, violations, **kwargs):
    return margins * tf.cast(violations, margins.dtype)


def bso_margins(logits, gold_beam, gold_ids, pred_ids, batch_size, beam_size):
    """
    Args:
        logits: shape = [batch, beam, vocab]
        gold_beam: shape = [batch] id of the beam whose previous token was the gold token
        gold_ids: shape = [batch] id of the gold token at this time step
        pred_ids: shape = [batch, beam] ids predicted by the beam search

    """
    # shape = [batch]
    worst_beam = (beam_size - 1) * tf.ones(shape=[batch_size], dtype=tf.int32)
    worst_ids = pred_ids[:, -1]
    worst_scores = get_entry(logits, worst_beam, worst_ids, batch_size, beam_size)

    # shape = [batch]
    gold_scores = get_entry(logits, gold_beam, gold_ids, batch_size, beam_size)
    bso_margins = worst_scores + 1 - gold_scores

    return bso_margins


def get_entry(t, indices_1d, indices_2d, batch_size, beam_size):
    """
    Args:
        t: shape = [batch, beam, vocab]
        indices_1d: shape = [batch]
        indices_2d: shape = [batch]

    Returns:
        o: shape = [batch], with o[i] = t[i, indices_1d[i], indices_2d[i]]

    """
    vocab_size = t.shape[2].value
    # shape = [batch]
    _indices = tf.range(batch_size) * beam_size * vocab_size
    _indices += indices_1d * vocab_size + indices_2d
    # shape = [batch * beam * vocab]
    t = tf.reshape(t, [-1])
    # shape = [batch]
    return tf.gather(t, _indices)


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
        self._state_size = BSOState(tf.TensorShape(1), tf.TensorShape(1), self._bs_cell.state_size,
                                    self._bs_cell.inputs_size, self._bs_cell.finished_size)
        self._output_size = self._bs_cell.output_size
        self._mistake_function = mistake_function
        self._batch_size = self._bs_cell._batch_size
        self._beam_size = self._bs_cell._beam_size
        self.initial_state = self._initial_state()


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


    def _initial_state(self):
        restarts = tf.ones(shape=[self._batch_size], dtype=tf.bool)
        gold_beam = tf.zeros(shape=[self._batch_size], dtype=tf.int32)
        beam_state, embeddings, finished = self._bs_cell.initialize()

        return BSOState(restarts, gold_beam, beam_state, embeddings, finished)


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
        beam_output, beam_state, embeddings, finished = self._bs_cell.step(state.restarts,
                state.beam_state, state.embeddings, state.finished)

        # 2. record violation in the loss
        margins = bso_margins(beam_output.logits, state.gold_beam, inputs.ids, beam_output.ids,
                self._batch_size, self._beam_size)
        violations = tf.greater(margins, tf.constant(0, dtype=margins.dtype))
        losses = self._mistake_function(logits=beam_output.logits, pred_ids=beam_output.ids,
                                        gold_ids=inputs.ids, margins=margins, violations=violations)

        # shape = [batch, beam]
        gold_ids = tf.tile(tf.expand_dims(inputs.ids, axis=1), [1, self._beam_size])
        # shape = [batch, beam]
        gold_in_beams = tf.equal(gold_ids, beam_output.ids)
        # shape = [batch]
        gold_in_beam = tf.reduce_any(gold_in_beams, axis=-1)
        gold_not_in_beam = tf.logical_not(gold_in_beam)
        restarts = tf.logical_or(gold_not_in_beam, violations)
        # shape = [batch]
        gold_beam = tf.argmax(tf.cast(gold_in_beams, tf.int32), axis=-1, output_type=tf.int32)

        # TODO(guillaume): adapt the state of the beam search and decoder so that they restart
        # shape = [batch, beam, embeddings]
        gold_beam = tf.where(restarts, self.initial_state.gold_beam, gold_beam)
        gold_embeddings = tf.tile(tf.expand_dims(inputs.embeddings, axis=1),
                [1, self._beam_size, 1])
        embeddings = tf.where(restarts, embeddings, gold_embeddings)

        new_output = BSOOutput(losses, beam_output.logits, beam_output.ids, beam_output.parents)
        new_state  = BSOState(restarts, gold_beam, beam_state, embeddings, finished)

        return (new_output, new_state)


    def __call__(self, inputs, state):
        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)


