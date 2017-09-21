from .config import Config


class Restore(Config):
    n_epochs = 2

    # learning rate schedule
    lr_init       = 1e-4
    lr_min        = 1e-4
    end_decay     = None # no decay
    end_warm      = None # no warm up
