from .config import Config


class Restore(Config):
    n_epochs = 2

    # learning rate schedule
    lr_init       = 1e-4
    lr_min        = 1e-4
    end_warm      = 0 # no warm up
