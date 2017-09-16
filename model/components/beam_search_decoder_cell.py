import tensorflow as tf
import collections
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell
from dynamic_decode import transpose_batch_time
from greedy_decoder_cell import DecoderOutput


class BeamSearchDecoderCellState(collections.namedtuple("BeamSearchDecoderCellState", 
                       ("cell_state", "log_probs"))):
    """
    State of the Beam Search decoding

    cell_state: shape = structure of [batch_size, beam_size, ?] 
        cell state for all the hypotheses
    embedding: shape = [batch_size, beam_size, embedding_size]
        embeddings of the previous time step for each hypothesis
    log_probs: shape = [batch_size, beam_size]
        log_probs of the hypotheses
    finished: shape = [batch_size, beam_size]
        boolean to know if one beam hypothesis has reached token id_END
    """
    pass


class BeamSearchDecoderOutput(collections.namedtuple("BeamSearchDecoderOutput", 
                        ("logits", "ids", "parents"))):
    """
    Stores the logic for the beam search decoding

    logits: shape = [batch_size, beam_size, vocab_size]
        scores before softmax of the beam search hypotheses
    ids: shape = [batch_size, beam_size] 
        ids of the best words at this time step
    parents: shape = [batch_size, beam_size] 
        ids of the beam index from previous time step
    """
    pass


class BeamSearchDecoderCell(object):
    def __init__(self, embeddings, cell, batch_size, start_token, beam_size, end_token):
        self._embeddings = embeddings
        self._cell = cell
        self._dim_embeddings = embeddings.shape[-1].value
        self._batch_size = batch_size
        self._start_token = start_token
        self._beam_size  = beam_size
        self._end_token = end_token
        self._vocab_size = embeddings.shape[0].value


    @property
    def output_dtype(self):
        """
        Needed for the custom dynamic_decode for the TensorArray of results
        """
        return BeamSearchDecoderOutput(
            logits=self._cell.output_dtype,
            ids=tf.int32,
            parents=tf.int32)


    @property
    def final_output_dtype(self):
        """
        For the finalize method
        """
        return DecoderOutput(
            logits=self._cell.output_dtype,
            ids=tf.int32)


    @property
    def state_size(self):
        return BeamSearchDecoderOutput(
            logits=tf.TensorShape([self._beam_size, self._vocab_size]),
            ids=tf.TensorShape([self._beam_size]),
            parents=tf.TensorShape([self._beam_size]))


    @property
    def final_output_size(self):
        return DecoderOutput(
            logits=tf.TensorShape([self._beam_size, self._vocab_size]),
            ids=tf.TensorShape([self._beam_size]))


    def initial_state(self):
        """
        Return initial state for the decoder
        """
        # cell initial state
        cell_state = self._cell.initial_state()
        cell_state = nest.map_structure(lambda t: tile_beam(t, self._beam_size), 
                                        cell_state)

        # prepare other initial states
        log_probs =  tf.zeros([self._batch_size, self._beam_size],
            dtype=self._cell.output_dtype)

        return BeamSearchDecoderCellState(
            cell_state,
            log_probs)


    def initial_inputs(self):
        return tf.tile(tf.reshape(self._start_token, [1, 1, self._dim_embeddings]),
                multiples=[self._batch_size, self._beam_size, 1])


    def initialize(self):
        initial_state = self.initial_state()
        initial_inputs = self.initial_inputs()
        initial_finished = tf.zeros(shape=[self._batch_size, self._beam_size], dtype=tf.bool)
        return initial_state, initial_inputs, initial_finished


    def step(self, time, state, embedding, finished):
        """
        Args:
            time: tensorf or int
            embedding: shape [batch_size, beam_size, d]
            state: structure of shape [bach_size, beam_size, ...]
            finished: structure of shape [batch_size, beam_size, ...]
        """
        # merge batch and beam dimension before callling step of cell
        cell_state = nest.map_structure(merge_batch_beam, state.cell_state)
        embedding = merge_batch_beam(embedding)

        # compute new logits
        logits, new_cell_state = self._cell.step(embedding, cell_state)

        # split batch and beam dimension before beam search logic
        new_logits = split_batch_beam(logits, self._beam_size)
        new_cell_state = nest.map_structure(
            lambda t: split_batch_beam(t, self._beam_size),
                new_cell_state)

        # compute log probs of the step
        # shape = [batch_size, beam_size, vocab_size]
        step_log_probs = tf.nn.log_softmax(new_logits)
        # shape = [batch_size, beam_size, vocab_size]
        step_log_probs = mask_probs(step_log_probs, self._end_token, finished)
        # shape = [batch_size, beam_size, vocab_size]
        log_probs = tf.expand_dims(state.log_probs, axis=-1) + step_log_probs

        # compute the best beams
        # shape =  (batch_size, beam_size * vocab_size)
        log_probs_flat = tf.reshape(log_probs, [self._batch_size, self._beam_size * self._vocab_size])
        log_probs_flat = tf.cond(
            time > 0,
            lambda: log_probs_flat,
            lambda: log_probs[:, 0])
        new_probs, indices = tf.nn.top_k(log_probs_flat, self._beam_size)

        # of shape [batch_size, beam_size]
        new_ids = indices % self._vocab_size
        new_parents = indices // self._vocab_size

        # get ids of words predicted and get embedding
        new_embedding = tf.nn.embedding_lookup(self._embeddings, new_ids)

        # compute end of beam
        finished = gather_helper(finished, new_parents, self._batch_size, self._beam_size)
        new_finished = tf.logical_or(finished, tf.equal(new_ids, self._end_token))

        new_cell_state = nest.map_structure(
            lambda t: gather_helper(t, new_parents, self._batch_size, self._beam_size),
            new_cell_state)


        # create new state of decoder
        new_state  = BeamSearchDecoderCellState(
            cell_state=new_cell_state,
            log_probs=new_probs)
        
        new_output = BeamSearchDecoderOutput(
            logits=new_logits, 
            ids=new_ids, 
            parents=new_parents)
        
        return (new_output, new_state, new_embedding, new_finished)


    def finalize(self, final_outputs, final_state):
        """
        Args:
            final_outputs: structure of tensors of shape 
                    [time dimension, batch_size, beam_size, d]
            final_state: instance of BeamSearchDecoderOutput

        Returns:
            [time, batch, beam, ...] structure of Tensor
        """        
        # reverse the time dimension
        maximum_iterations = tf.shape(final_outputs.ids)[0]
        final_outputs = nest.map_structure(
            lambda t: tf.reverse(t, axis=[0]), final_outputs)
        
        # initial states
        def create_ta(d):
            return tf.TensorArray(
                dtype=d,
                size=maximum_iterations)

        initial_time = tf.constant(0, dtype=tf.int32)
        initial_outputs_ta = nest.map_structure(create_ta, self.final_output_dtype)
        initial_parents = tf.tile(
            tf.expand_dims(tf.range(self._beam_size), axis=0),
            multiples=[self._batch_size, 1])

        def condition(time, outputs_ta, parents):
            return tf.less(time, maximum_iterations)

        # beam search decoding cell
        def body(time, outputs_ta, parents):
            # get ids, logits and parents predicted at this time step by decoder
            input_t = nest.map_structure(
                lambda t: t[time], final_outputs)

            # extract the entries corresponding to parents
            new_state = nest.map_structure(
                lambda t: gather_helper(t, parents, self._batch_size, self._beam_size),
                input_t)

            # create new output
            new_output = DecoderOutput(
                logits=new_state.logits,
                ids=new_state.ids)

            # write beam ids
            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                      outputs_ta, new_output)

            return (time + 1), outputs_ta, parents
        
        res = tf.while_loop(
            condition,
            body,
            loop_vars=[initial_time, initial_outputs_ta, initial_parents])

        # unfold and stack the structure from the nested tas
        final_outputs = nest.map_structure(lambda ta: ta.stack(), res[1])

        # reverse time step
        final_outputs = nest.map_structure(
            lambda t: tf.reverse(t, axis=[0]), final_outputs)

        return DecoderOutput(
            logits=final_outputs.logits,
            ids=final_outputs.ids)


def merge_batch_beam(t):
    """
    Args:
        t: tensor of shape [batch_size, beam_size, ...]
            whose dimensions after beam_size must be statically known

    Returns:
        t: tensorf of shape [batch_size * beam_size, ...]
    """
    batch_size = tf.shape(t)[0]
    beam_size = t.shape[1].value

    if t.shape.ndims == 2:
        return tf.reshape(t, [batch_size*beam_size, 1])
    elif t.shape.ndims == 3:
        return tf.reshape(t, [batch_size*beam_size, t.shape[-1].value])
    elif t.shape.ndims == 4:
        return tf.reshape(t, [batch_size*beam_size, t.shape[-2].value, t.shape[-1].value])
    else:
        raise NotImplementedError


def split_batch_beam(t, beam_size):
    """
    Args:
        t: tensorf of shape [batch_size*beam_size, ...]

    Returns:
        t: tensor of shape [batch_size, beam_size, ...]
    """
    if t.shape.ndims == 1:
        return tf.reshape(t, [-1, beam_size])
    elif t.shape.ndims == 2:
        return tf.reshape(t, [-1, beam_size, t.shape[-1].value])
    elif t.shape.ndims == 3:
        return tf.reshape(t, [-1, beam_size, t.shape[-2].value, t.shape[-1].value])
    else:
        raise NotImplementedError


def tile_beam(t, beam_size):
    """
    Args:
        t: tensor of shape [batch_size, ...]

    Returns:
        t: tensorf of shape [batch_size, beam_size, ...]
    """
    # shape = [batch_size, 1 , x]
    t = tf.expand_dims(t, axis=1)
    if t.shape.ndims == 2:
        multiples = [1, beam_size]
    elif t.shape.ndims == 3:
        multiples = [1, beam_size, 1]
    elif t.shape.ndims == 4:
        multiples = [1, beam_size, 1, 1]

    return tf.tile(t, multiples)


def mask_probs(probs, end_token, finished):
    """
    Args:
        probs: tensor of shape [batch_size, beam_size, vocab_size]
        end_token: (int)
        finished: tensor of shape [batch_size, beam_size], dtype = tf.bool
    """
    # one hot of shape [vocab_size]
    vocab_size = probs.shape[-1].value
    one_hot = tf.one_hot(end_token, vocab_size,
        on_value=0.,
        off_value=probs.dtype.min,
        dtype=probs.dtype)
    # expand dims of shape [batch_size, beam_size, 1]
    finished = tf.expand_dims(tf.cast(finished, probs.dtype), axis=-1)

    return (1. - finished) * probs + finished * one_hot


def gather_helper(t, indices, batch_size, beam_size):
    """
    Args:
        t: tensor of shape = [batch_size, beam_size, d]
        indices: tensor of shape = [batch_size, beam_size]

    Returns:
        new_t: tensor of same shape as t but new_t[:, i] = t[:, new_parents[:, i]]
    """
    range_  = tf.expand_dims(tf.range(batch_size) * beam_size, axis=1);
    indices = tf.reshape(indices + range_, [-1])
    output  = tf.gather(
        tf.reshape(t, [batch_size*beam_size, -1]),
        indices)
    
    if t.shape.ndims == 2:
        return tf.reshape(output, [batch_size, beam_size])

    elif t.shape.ndims == 3:
        d = t.shape[-1].value
        return tf.reshape(output, [batch_size, beam_size, d])