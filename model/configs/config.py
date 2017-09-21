import os
from shutil import copyfile


from ..utils.data import load_tok_to_id, PAD, END
from ..utils.general import get_logger, init_dir, init_file
from .load import load_config_json, init_directories


class Config():
    """Class that loads hyperparameters from json file"""

    def __init__(self, jsons=None, path_vocab=None):
        """
        Args:
            jsons: list of path_to_json files
            path_vocab: path

        """
        if jsons is not None:
            for path_json in jsons:
                self.load_config(path_json)
                copyfile(path_json, self.dir_output + DEFAULT_CONFIG)
            self.load_logger()

        if path_vocab is not None:
            self.load_vocab(path_vocab)
            copyfile(path_vocab, self.dir_output + DEFAULT_VOCAB)


    def load_config(self, path_json):
        json_config = load_config_json(path_json)
        self.__dict__.update(json_config)
        init_directories(json_config)


    def load_logger(self):
        self.logger = get_logger(self.dir_output + DEFAULT_LOG)


    def load_vocab(self, path_vocab=None):
        path_vocab = self.path_vocab if path_vocab is None
        self.tok_to_id = load_tok_to_id(path_vocab)
        self.id_to_tok = {idx: tok for tok, idx in self.tok_to_id.iteritems()}
        self.n_tok = len(self.tok_to_id)

        self.attn_cell_config["num_proj"] = self.n_tok
        self.id_PAD = self.tok_to_id[PAD]
        self.id_END = self.tok_to_id[END]


    def restore_from_dir(self, dir_output):
        path_json = dir_output + DEFAULT_CONFIG
        self.load_config(path_json)

        self.dir_output = dir_output # overwrite
        self.load_logger()

        path_vocab = dir_output + DEFAULT_VOCAB
        self.load_vocab(path_vocab)


