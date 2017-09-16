import sys
import numpy as np
import time
import tensorflow as tf
import tensorflow.contrib.layers as layers


from .utils.general import Progbar
from .utils.data import minibatches, pad_batch_images, \
    pad_batch_formulas, load_vocab
from .utils.eval import write_answers, evaluate


from .encoder import Encoder
from .decoder import Decoder
from .base import BaseModel


class Img2SeqModel(BaseModel):
    """Specialized class for Img2Seq Model"""

    def build(self):
        """Builds model"""
        self.logger.info("Building model...")

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

        self._add_placeholders_op()
        self._add_pred_op()
        self._add_loss_op()

        self._build()

        self.logger.info("- done.")


    def _add_placeholders_op(self):
        """
        Add placeholder attributes
        """
        # hyper params
        self.lr = tf.placeholder(tf.float32, shape=(),
            name='lr')
        self.dropout = tf.placeholder(tf.float32, shape=(),
            name='dropout')
        self.training = tf.placeholder(tf.bool, shape=(),
            name="training")


        # input of the graph
        self.img = tf.placeholder(tf.uint8, shape=(None, None, None, 1),
            name='img')
        self.formula = tf.placeholder(tf.int32, shape=(None, None),
            name='formula')
        self.formula_length = tf.placeholder(tf.int32, shape=(None, ),
            name='formula_length')

        # for tensorboard
        tf.summary.scalar("lr", self.lr)


    def _get_feed_dict(self, img, training, formula=None, lr=None, dropout=1):
        """Returns a dict"""
        img = pad_batch_images(img)

        fd = {
            self.img: img,
            self.dropout: dropout,
            self.training: training,
        }

        if formula is not None:
            formula, formula_length = pad_batch_formulas(formula,
                    self.config.id_PAD, self.config.id_END)
            # print img.shape, formula.shape
            fd[self.formula] = formula
            fd[self.formula_length] = formula_length
        if lr is not None:
            fd[self.lr] = lr

        return fd


    def _add_pred_op(self):
        """Defines self.pred"""
        encoded_img = self.encoder(self.training, self.img, self.dropout)
        train, test = self.decoder(self.training, encoded_img, self.formula,
                self.dropout)

        self.pred_train = train
        self.pred_test  = test


    def _add_loss_op(self):
        """Defines self.loss"""
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.pred_train, labels=self.formula)

        mask = tf.sequence_mask(self.formula_length)
        losses = tf.boolean_mask(losses, mask)

        # loss for training
        self.loss = tf.reduce_mean(losses)

        # # to compute perplexity for test
        self.ce_words = tf.reduce_sum(losses) # sum of CE for each word
        self.n_words = tf.reduce_sum(self.formula_length) # number of words

        # for tensorboard
        tf.summary.scalar("loss", self.loss)



    def _run_epoch(self, train_set, val_set, lr_schedule):
        """Performs an epoch of training

        Args:
            train_set: Dataset instance
            val_set: Dataset instance
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            score: (float) model will select weights that achieve the highest
                score

        """
        # logging
        tic = time.time()
        batch_size = self.config.batch_size
        nbatches = (len(train_set) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
            # get feed dict
            fd = self._get_feed_dict(img, training=True, formula=formula,
                    lr=lr_schedule.lr, dropout=self.config.dropout)

            # update step
            _, loss_eval, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)
            prog.update(i + 1, [("loss", loss_eval), ("perplexity",
                    np.exp(loss_eval))])

             # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

            # update learning rate
            lr_schedule.update(batch_no=epoch*nbatches + i)

        # logging
        toc = time.time()
        self.logger.info("- Epoch {} - time: {:04.2f}, lr: {:04.5f}".format(
                        epoch, toc-tic, lr_schedule.lr))
        # evaluation
        metrics = self.evaluate(val_set)
        score = metrics[self.config.metric_val] # selection

        lr_schedule.update(score=score) # for early stopping

        return score


    def _run_evaluate(self, test_set):
        """Performs an epoch of evaluation

        Args:
            test_set: Dataset instance

        Returns:
            metrics: (dict) metrics["acc"] = 0.85 for instance

        """
        references, hypotheses = [], []
        n_words, ce_words = 0, 0 # sum of ce for all words + nb of words

        for img, formula in minibatches(val_set, self.config.batch_size):
            fd = self._get_feed_dict(img, training=False, formula=formula,
                    dropout=1)
            ce_words_eval, n_words_eval, ids_eval = self.sess.run(
                    [self.ce_words, self.n_words, self.pred_test.ids],
                    feed_dict=fd)

            if self.config.decoding == "greedy":
                ids_eval = np.expand_dims(ids_eval, axis=1)

            elif self.config.decoding == "beam_search":
                ids_eval = np.transpose(ids_eval, [0, 2, 1])

            n_words += n_words_eval
            ce_words += ce_words_eval
            for form, pred in zip(formula, ids_eval):
                # pred is of shape (number of hypotheses, time)
                references.append([form])
                hypotheses.append(pred)


        scores = evaluate(references, hypotheses, self.config.id_to_tok,
                self.config.path_results, self.config.id_END)

        ce_mean = ce_words / float(n_words)
        scores["perplexity"] = np.exp(ce_mean)

        return scores
