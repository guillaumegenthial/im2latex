from .config import Config


class Small(Config):
    """Class for testing"""
    dir_output = "results/small/"

    path_vocab    = "data/small_vocab.txt"
    min_count_tok = 2

    dir_images_train = "data/small/"
    dir_images_test  = "data/small/"
    dir_images_val   = "data/small/"

    path_matching_train = "data/small.matching.txt"
    path_matching_val   = "data/small.matching.txt"
    path_matching_test  = "data/small.matching.txt"

    path_formulas_train = "data/small.formulas.norm.txt"
    path_formulas_test  = "data/small.formulas.norm.txt"
    path_formulas_val   = "data/small.formulas.norm.txt"

    n_epochs = 2
    batch_size = 3
    max_iter = 10
    decoding = "beam_search"
    encode_with_lstm = False

    lr_init       = 1e-3
    lr_min        = 1e-3