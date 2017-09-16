import numpy as np


class LRSchedule(object):
    def __init__(self, lr_init=1e-3, lr_min=1e-4, start_decay=0, decay_rate=None, end_decay=None,
        lr_warm=1e-4, end_warm=None):
        # store parameters
        self.lr_init     = lr_init
        self.lr_min      = lr_min
        self.start_decay = start_decay # id of batch to start decay
        self.decay_rate  = decay_rate # optional: if provided, decay if no improval
        self.end_decay   = end_decay # optional: if provided, exp decay
        self.lr_warm     = lr_warm
        self.end_warm    = end_warm # optional: if provided, warm start

        self.early_stopping = False

        if self.end_decay is not None:
            self.exp_decay = np.power(lr_min/lr_init, 1/float(end_decay - start_decay))

        # initialize learning rate and score on eval
        self.score = None
        self.lr    = lr_init

        # warm start initializes learning rate to warm start
        if self.end_warm is not None:
            self.lr = self.lr_warm


    def update(self, batch_no=None, score=None):
        """
        Update the learning rate:
            - decay by self.decay rate if score is lower than previous best
            - decay by self.decay_rate

        Both updates can concurrently happen if both
            - self.decay_rate is not None
            - self.n_steps is not None
        """
        # update based on time
        if batch_no is not None:
            if self.end_warm is not None and self.end_warm < batch_no < self.start_decay:
                self.lr = self.lr_init

            if batch_no > self.start_decay and self.end_decay is not None:
                self.lr *= self.exp_decay

        # update based on performance
        if self.decay_rate is not None:
            if score is not None and self.score is not None:
                # assume lower is better
                if score > self.score:
                    self.lr *= self.decay_rate

        # update last score eval
        if score is not None:
            self.score = score

        self.lr = max(self.lr, self.lr_min)
