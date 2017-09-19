from .config import Config


class Test(Config):
    """Class for testing"""
    n_epochs = 2
    batch_size = 10
    max_iter = 10
    max_length_formula = 20
    decoding = "beam_search"
    encode_with_lstm = False
