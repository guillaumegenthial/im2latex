import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


from .utils.general import Progbar
from .utils.data import minibatches, pad_batch_images, pad_batch_formulas
from .evaluation.util import write_answers
from .evaluation.text import score_files


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

        self._build() # train op and init session

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

        # tensorboard
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



    def _run_epoch(self, train_set, val_set, epoch, lr_schedule):
        """Performs an epoch of training

        Args:
            train_set: Dataset instance
            val_set: Dataset instance
            epoch: (int) id of the epoch, starting at 0
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            score: (float) model will select weights that achieve the highest
                score

        """
        # logging
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


        # evaluation
        scores = self.evaluate(val_set,
                {"dir_name": self.config.dir_formulas_val_result})
        score = scores[self.config.metric_val]
        lr_schedule.update(score=score)

        return score



    def write_prediction(self, test_set, params):
        """Performs an epoch of evaluation

        Args:
            test_set: Dataset instance
            params: (dict) with extra params in it
                - "dir_name": (string)

        Returns:
            files: list of path to files
            perp: (float) perplexity on test set

        """
        # initialize containers of references and predictions
        if self.config.decoding == "greedy":
            refs, hyps = [], [[]]
        elif self.config.decoding == "beam_search":
            refs, hyps = [], [[] for i in range(self.config.beam_size)]

        # iterate over the dataset
        n_words, ce_words = 0, 0 # sum of ce for all words + nb of words
        for img, formula in minibatches(test_set, self.config.batch_size):
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
            for form, preds in zip(formula, ids_eval):
                refs.append(form)
                for i, pred in enumerate(preds):
                    hyps[i].append(pred)

        files = write_answers(refs, hyps, self.config.id_to_tok,
                params["dir_name"], self.config.id_END)

        perp = - np.exp(ce_words / float(n_words))

        return files, perp


    def _run_evaluate(self, test_set, params):
        """Performs an epoch of evaluation

        Args:
            test_set: Dataset instance
            params: (dict) with extra params in it
                - "dir_name": (string)

        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance

        """
        files, perp = self.write_prediction(test_set, params)
        scores = score_files(files[0], files[1])
        scores["perplexity"] = perp

        return scores
