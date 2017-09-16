from configs.config import Config


class Test(Config):
    """Class for testing"""

    n_epochs = 1
    batch_size = 20
    max_iter = 20
    max_length_formula = 50
    decoding = "beam_search"
    encode_with_lstm = False
