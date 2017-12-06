import collections
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell


from beam_search_decoder import BeamSearchDecoderOutput, BeamSearchDecoderCellState, tile_beam, \
    gather_helper


class BSOState(collections.namedtuple("BSOState",
    ("restarts", "gold_beam", "beam_state", "embeddings", "finished"))):
    """State for Beam Search Optimization wrapper

    restarts: (tf.bool) shape = [batch], if True, need to restart from restart_indices beam
    gold_beam: (tf.int32) shape = [batch], id of the beam that has the gold_beam
    beam_state: BeamSearchDecoderCellState
    embeddings: (tf.float32) shape = [batch, beam, embedding]
    finished: (tf.bool) shape = [batch, beam] true if beam reached the <eos> token

    """
    pass


class BSOOutput(collections.namedtuple("BSOOutput",
    ("losses", "logits", "ids", "parents"))):
    """Output of BSO = BeamSearchOutput + losses

    losses: (tf.float32) shape = [batch]
    logits: shape = [batch_size, beam_size, vocab_size]
        scores before softmax of the beam search hypotheses
    ids: shape = [batch_size, beam_size]
        ids of the best words at this time step
    parents: shape = [batch_size, beam_size]
        ids of the beam index from previous time step

    """
    pass


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
    # TODO(guillaume): different loss for last time step
    return margins * tf.cast(violations, margins.dtype)


def bso_margins(logits, gold_beam, gold_ids, pred_beam, pred_ids):
    """
    Args:
        logits: shape = [batch, beam, vocab]
        gold_beam: shape = [batch] id of the beam whose previous token was the gold token
        gold_ids: shape = [batch] id of the gold token at this time step
        pred_beam: shape = [batch, beam] id of the beam whose predicted token is attached
        pred_ids: shape = [batch, beam] ids predicted by the beam search

    """
    # variables
    beam_size = logits.shape[1].value
    batch_size = tf.shape(logits)[0]
    # extract K-th hypothesis score
    worst_beam = pred_beam[:, -1]
    worst_ids = pred_ids[:, -1]
    worst_scores = get_entry(logits, worst_beam, worst_ids)
    # extract gold hypothesis score
    gold_scores = get_entry(logits, gold_beam, gold_ids)
    bso_margins = worst_scores + 1 - gold_scores

    return bso_margins


def get_entry(t, indices_1d, indices_2d):
    """
    Args:
        t: shape = [batch, beam, vocab]
        indices_1d: shape = [batch]
        indices_2d: shape = [batch]

    Returns:
        o: shape = [batch], with o[i] = t[i, indices_1d[i], indices_2d[i]]

    """
    batch_size = tf.shape(t)[0]
    indices = tf.stack([tf.range(batch_size), indices_1d, indices_2d], axis=1)
    return tf.gather_nd(t, indices)


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


    def step(self, gold_ids, state):
        """
        Args:
            gold_ids: token at time step t, ids: shape = [batch_size]
                embeddings of gold words from time step t
            state: instance of BSOState from previous time step t-1

        """
        # 1. get id of (t-1) gold beam and (t) gold ids
        # shape = [batch]
        gold_beam = state.gold_beam
        # shape = [batch, beam]
        gold_beam_tiled = tile_beam(gold_beam, self._beam_size)
        gold_ids_tiled  = tile_beam(gold_ids, self._beam_size)

        # 2. expand beam with regular beam search
        o = self._bs_cell._step(state.restarts, state.beam_state, state.embeddings, state.finished)
        (beam_output, beam_state, embeddings, finished, exp_cell_state, exp_log_probs) = o

        # 3. beam from gold beam to gold_id
        gold_embeddings, gold_finished, gold_cell_state = self._bs_cell._gather(gold_ids_tiled,
                gold_beam_tiled, exp_cell_state, state.finished)
        # shape = [batch]
        gold_log_probs = get_entry(exp_log_probs, gold_beam, gold_ids)
        gold_log_probs = tile_beam(gold_log_probs, self._beam_size)
        gold_beam_state = BeamSearchDecoderCellState(cell_state=gold_cell_state,
                                                     log_probs=gold_log_probs)

        # 4. compute margins and violations
        logits, pred_beam, pred_ids = (beam_output.logits, beam_output.parents, beam_output.ids)
        # shape = [batch]
        margins = bso_margins(logits, gold_beam, gold_ids, pred_beam, pred_ids)
        violations = tf.greater(margins, tf.constant(0, dtype=margins.dtype))
        losses = self._mistake_function(logits=logits, pred_ids=pred_ids, gold_ids=gold_ids,
                margins=margins, violations=violations)

        # 5. check if gold hypothesis fell out of the beam
        # shape = [batch, beam]
        gold_ids_in_beams = tf.equal(gold_ids_tiled, pred_ids)
        # shape = [batch]
        gold_in_beam = tf.reduce_any(gold_ids_in_beams, axis=-1)
        gold_not_in_beam = tf.logical_not(gold_in_beam)

        # 6. compute next state, need to restart from gold if margin violation or gold fell
        # shape = [batch]
        new_restarts = tf.logical_or(gold_not_in_beam, violations)
        new_gold_beam = tf.argmax(tf.cast(gold_ids_in_beams, tf.int32), axis=-1,
                                  output_type=tf.int32)


        # shape = [batch, beam, embeddings]
        apply_restart = lambda t, gold_t: tf.where(new_restarts, gold_t, t)
        new_gold_beam = apply_restart(new_gold_beam, self.initial_state.gold_beam)
        new_embeddings = apply_restart(embeddings, gold_embeddings)
        new_beam_state = nest.map_structure(apply_restart, beam_state, gold_beam_state)
        new_finished = apply_restart(finished, gold_finished)

        new_output = BSOOutput(losses, beam_output.logits, beam_output.ids, beam_output.parents)
        new_state  = BSOState(new_restarts, new_gold_beam, new_beam_state, new_embeddings, new_finished)

        return (new_output, new_state)


    def __call__(self, inputs, state):
        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)


