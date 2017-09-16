import os
import shutil


from utils.data import load_tok_to_id, PAD, END
from utils.general import get_logger, init_dir, init_file


class Config():

    def __init__(self):
        """Creates output directories if they don't exist and load vocabulary

        Defines attributes that depends on the vocab.
        Look for the __init__ comments in the class attributes
        """
        # directory for training outputs
        init_dir(self.dir_output)
        init_dir(self.dir_model)
        init_dir(self.dir_plots)

        # initializer file for answers
        init_file(self.path_results)
        init_file(self.path_results_final)

        # load vocabs
        self.tok_to_id = load_tok_to_id(self.path_vocab)
        self.id_to_tok = {idx: tok for tok, idx in tok_to_id.iteritems()}
        self.n_tok = len(self.tok_to_id)

        self.attn_cell_config["num_proj"] = self.ntok
        self.id_PAD = self.tok_to_id[PAD]
        self.id_END = self.tok_to_id[END]

        self.logger = get_logger(self.path_log)


    # directories
    dir_output = "results/full_eval_beam/"
    dir_images = "../data/images_processed"
    dir_plots  = dir_output + "plots/"
    dir_model  = dir_output + "model.weights/"

    # paths
    path_vocab          = "../data/latex_vocab.txt"
    path_log            = dir_output + "log.txt"
    path_results        = dir_output + "results_val.txt"
    path_results_final  = dir_output + "results.txt"
    path_results_img    = dir_output + "images/"

    # training data
    path_matching_train = "../data/train_filter.lst"
    path_matching_val   = "../data/val_filter.lst"
    path_matching_test  = "../data/test_filter.lst"
    path_formulas       = "../data/norm.formulas.lst"

    max_length_formula = 150
    max_iter      = None

    # encoder
    encoder_dim = 256
    encode_with_lstm = False
    encode_mode = "vanilla"

    # decoder
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

    path_embeddings = "../data/embeddings.npz"
    pretrained_embeddings = False
    trainable_embeddings = True
    dim_embeddings = 80

    # training parameters
    lr_method     = "Adam"
    n_epochs      = 15
    batch_size    = 20
    dropout       = 1 # keep_prob
    metric_val    = "perplexity"

    # learning rate schedule
    lr_init       = 1e-3
    lr_min        = 1e-5
    start_decay   = 9 # start decaying
    end_decay     = 13 # end decay
    decay_rate    = 0.5 # decay rate if perf does not improve
    lr_warm       = 1e-4 # warm up with lower learning rate because of high gradients
    end_warm      = 2 # keep warm up for 2 epochs



