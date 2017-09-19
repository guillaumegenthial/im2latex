import os
import shutil


from ..utils.data import load_tok_to_id, PAD, END
from ..utils.general import get_logger, init_dir, init_file


class Config():

    def __init__(self, load=True):
        """Creates output directories if they don't exist and load vocabulary

        Defines attributes that depends on the vocab.
        Look for the __init__ comments in the class attributes
        """
        # result directories
        dir_output      = self.dir_output
        self.dir_plots  = dir_output + "plots/"
        self.dir_model  = dir_output + "model.weights/"

        # result paths
        self.path_log                  = dir_output + "config.log"
        self.path_results_img          = dir_output + "images/"
        self.path_formulas_val_result  = dir_output + "val.formulas.result.txt"
        self.path_formulas_test_result = dir_output + "test.formulas.result.txt"

        # directory for training outputs
        init_dir(self.dir_output)
        init_dir(self.dir_model)
        init_dir(self.dir_plots)

        # initializer file for answers
        init_file(self.path_formulas_test_result)
        init_file(self.path_formulas_val_result)

        # load vocabs
        if load:
            self.tok_to_id = load_tok_to_id(self.path_vocab)
            self.id_to_tok = {idx: tok for tok, idx in
                              self.tok_to_id.iteritems()}
            self.n_tok = len(self.tok_to_id)

            self.attn_cell_config["num_proj"] = self.n_tok
            self.id_PAD = self.tok_to_id[PAD]
            self.id_END = self.tok_to_id[END]

        self.logger = get_logger(self.path_log)


    # root directory for results
    dir_output = "results/full_eval_beam/"

    # data
    dir_images_train = "data/images_train/"
    dir_images_test  = "data/images_test/"
    dir_images_val   = "data/images_val/"

    path_matching_train = "data/train.matching.txt"
    path_matching_val   = "data/val.matching.txt"
    path_matching_test  = "data/test.matching.txt"

    path_formulas_train = "data/train.formulas.norm.txt"
    path_formulas_test  = "data/test.formulas.norm.txt"
    path_formulas_val   = "data/val.formulas.norm.txt"

    # vocab
    path_vocab         = "data/vocab.txt"
    min_count_tok      = 10
    max_length_formula = 150
    max_iter           = None

    # encoder
    encoder_dim = 256
    encode_with_lstm = False
    encode_mode = "vanilla"

    # decoder
    path_embeddings = "data/embeddings.npz"
    pretrained_embeddings = False
    trainable_embeddings = True
    dim_embeddings = 80

    attn_cell_config = {
        "cell_type": "lstm",
        "num_units": 512,
        "dim_e": 512,
        "dim_o": 512,
        "num_proj": None, # to be computed in __init__  because vocab size
        "dim_embeddings": dim_embeddings
    }
    decoding = "greedy" # "greedy" or "beam_search"
    beam_size = 5

    # training parameters
    lr_method     = "Adam"
    n_epochs      = 15
    batch_size    = 20
    dropout       = 1 # keep_prob
    metric_val    = "perplexity"
    clip          = -1

    # learning rate schedule
    lr_init       = 1e-3
    lr_min        = 1e-5
    start_decay   = 9 # start decaying
    end_decay     = 13 # end decay
    decay_rate    = 0.5 # decay rate if perf does not improve
    lr_warm       = 1e-4 # warm up: small lr because of high gradients
    end_warm      = 2 # keep warm up for 2 epochs
