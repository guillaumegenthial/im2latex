from utils.dataset import Dataset
from models.model import Model
from configs.config import Config, Test
from utils.preprocess import greyscale, get_form_prepro, compose
from utils.data_utils import minibatches, pad_batch_formulas, \
    pad_batch_images
from utils.lr_schedule import LRSchedule
import tensorflow as tf
from utils.evaluate import simple_plots, evaluate_dataset


if __name__ == "__main__":
    # Load config
    # config = Config()
    config = Test()

    path_formulas = config.path_results_final
    path_matching_eval = config.dir_output + "matching.txt"
    dir_images = config.dir_output + "images/"

    max_lengths = [20, 50, 75, 100, 150]
    all_scores = None

    for i, max_length in enumerate(max_lengths):
        config.logger.info("TEST: max-length = {}".format(max_length))

        # get dataset from the formulas outputed by our model
        test_set  =  Dataset(path_formulas=path_formulas, dir_images=dir_images,
                        path_matching=path_matching_eval, img_prepro=greyscale, 
                        form_prepro=get_form_prepro(config.vocab), max_len=max_length,
                        max_iter=10, bucket=False, single=False)


        # Build model
        scores = evaluate_dataset(test_set, config.dir_plots, prefix=str(max_length))
        scores_to_print = " - ".join(["{}: {:04.2f}".format(name, value) for name, value in scores.iteritems()])
        config.logger.info(scores_to_print)

        if all_scores is None:
            all_scores = dict()
            for k, v in scores.iteritems():
                all_scores[k] = [v]
        else:
            for k, v in scores.iteritems():
                all_scores[k].append(v)

    simple_plots(max_lengths, all_scores, config.dir_plots)

