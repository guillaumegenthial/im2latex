from utils.dataset import Dataset
from models.model import Model
from configs.config import Config, Test
from utils.preprocess import greyscale, get_form_prepro, compose
from utils.data_utils import minibatches, pad_batch_formulas, \
    pad_batch_images
from utils.lr_schedule import LRSchedule
import tensorflow as tf
from utils.evaluate import evaluate_dataset


if __name__ == "__main__":
    # Load config
    config = Config()

    path_formulas = config.path_results_final
    path_matching_eval = config.dir_output + "matching.txt"
    dir_images = config.dir_output + "images/"

    # get dataset from the formulas outputed by our model
    test_set  =  Dataset(path_formulas=path_formulas, dir_images=dir_images,
                    path_matching=path_matching_eval, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(config.vocab), max_len=config.max_length_formula,
                    max_iter=config.max_iter, bucket=False, single=False)

    # generate the images and the matching from the formulas
    # test_set.generate_from_formulas(single=False)

    # evaluate
    scores = evaluate_dataset(test_set, config.dir_plots)
    scores_to_print = " - ".join(["{}: {:04.2f}".format(name, value) for name, value in scores.iteritems()])
    config.logger.info(scores_to_print)
