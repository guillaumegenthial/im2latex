import numpy as np


from .data import UNK


def greyscale(state):
    """Preprocess state (:, :, 3) image into greyscale"""
    state = state[:, :, 0]*0.299 + state[:, :, 1]*0.587 + state[:, :, 2]*0.114
    state = state[:, :, np.newaxis]
    return state.astype(np.uint8)


def downsample(state):
    """Downsamples an image on the first 2 dimensions

    Args:
        state: (np array) with 3 dimensions

    """
    return state[::2, ::2, :]


def get_form_prepro(vocab):
    """Given a vocab, returns a lambda function word -> id

    Args:
        vocab: dict[token] = id

    Returns:
        lambda function(formula) -> list of ids

    """
    def get_token_id(token):
        return vocab[token] if token in vocab else vocab[UNK]

    def f(formula):
        formula = formula.strip().split(' ')
        return map(lambda t: get_token_id(t), formula)

    return f
