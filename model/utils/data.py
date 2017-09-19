import numpy as np
from PIL import Image
import time


UNK = "_UNK" # for unknown words
PAD = "_PAD" # for padding
END = "_END" # for the end of a caption


def render(arr):
    """
    Render an array as an image

    Args:
        arr: np array (np.uint8) representing an image

    """
    mode = "RGB" if arr.shape[-1] == 3 else "L"
    img =  Image.fromarray(np.squeeze(arr), mode)
    img.show()


def get_max_shape(arrays):
    """
    Args:
        images: list of arrays

    """
    shapes = map(lambda x: list(x.shape), arrays)
    ndim = len(arrays[0].shape)
    max_shape = []
    for d in range(ndim):
        max_shape += [max(shapes, key=lambda x: x[d])[d]]

    return max_shape


def pad_batch_images(images, max_shape=None):
    """
    Args:
        images: list of arrays
        target_shape: shape at which we want to pad

    """

    # 1. max shape
    if max_shape is None:
        max_shape = get_max_shape(images)

    # 2. apply formating
    batch_images = 255 * np.ones([len(images)] + list(max_shape))
    for idx, img in enumerate(images):
        batch_images[idx, :img.shape[0], :img.shape[1]] = img

    return batch_images.astype(np.uint8)


def pad_batch_formulas(formulas, id_PAD, id_END, max_len=None):
    """Pad formulas to the max length with id_PAD and adds and id_END token
    at the end of each formula

    Args:
        formulas: (list) of list of ints
        max_length: length maximal of formulas

    Returns:
        array: of shape = (batch_size, max_len) of type np.int32
        array: of shape = (batch_size) of type np.int32

    """
    if max_len is None:
        max_len = max(map(lambda x: len(x), formulas))

    batch_formulas = id_PAD * np.ones([len(formulas), max_len+1],
            dtype=np.int32)
    formula_length = np.zeros(len(formulas), dtype=np.int32)
    for idx, formula in enumerate(formulas):
        batch_formulas[idx, :len(formula)] = np.asarray(formula,
                dtype=np.int32)
        batch_formulas[idx, len(formula)]  = id_END
        formula_length[idx] = len(formula) + 1

    return batch_formulas, formula_length


def minibatches(data_generator, minibatch_size):
    """
    Args:
        data_generator: generator of (img, formulas) tuples
        minibatch_size: (int)

    Returns:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data_generator:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def load_tok_to_id(filename):
    """
    Args:
        filename: (string) path to vocab txt file one word per line

    Returns:
        dict: d[token] = id

    """
    tok_to_id = dict()
    with open(filename) as f:
        for idx, token in enumerate(f):
            token = token.strip()
            tok_to_id[token] = idx

    # add pad and unk tokens
    tok_to_id[PAD] = len(tok_to_id)
    tok_to_id[UNK] = len(tok_to_id)
    tok_to_id[END] = len(tok_to_id)

    return tok_to_id


def reconstruct_formula(tokens, rev_vocab):
    """
    Args:
        tokens: list of idx each comprised between 0 and len(vocab) - 1
        rev_vocab: dict such that rev_vocab[idx] = word

    Returns:
        string resulting from mapping of idx to words with rev_vocab and then
            concatenation

    """
    result = []
    for token in tokens:
        word = rev_vocab[token]
        result.append(word)

    return " ".join(result)
