from model.utils.data_generator import DataGenerator
from model.img2seq import Img2SeqModel
from model.configs.config import Config
from model.configs.test import Test
from model.utils.preprocess import greyscale, get_form_prepro, compose
from model.utils.data import minibatches, pad_batch_formulas, \
    pad_batch_images
from model.utils.lr_schedule import LRSchedule


if __name__ == "__main__":
    # Load config
    # config = Config()
    config = Test() # for test purposes

    # Load datasets
    train_set = DataGenerator(path_formulas=config.path_formulas,
            dir_images=config.dir_images, max_iter=config.max_iter,
            path_matching=config.path_matching_train, img_prepro=greyscale,
            form_prepro=get_form_prepro(config.tok_to_id),
            max_len=config.max_length_formula)

    val_set = DataGenerator(path_formulas=config.path_formulas,
            dir_images=config.dir_images, max_iter=config.max_iter,
            path_matching=config.path_matching_val, img_prepro=greyscale,
            form_prepro=get_form_prepro(config.tok_to_id),
            max_len=config.max_length_formula)


    train_set = val_set # test

    n_batches_epoch = ((len(train_set) + config.batch_size - 1) //
                        config.batch_size)

    lr_schedule = LRSchedule(lr_init=config.lr_init, lr_min=config.lr_min,
                            start_decay=config.start_decay*n_batches_epoch,
                            end_decay=config.end_decay*n_batches_epoch,
                            lr_warm=config.lr_warm,
                            end_warm=config.end_warm*n_batches_epoch)

    # Build model
    model = Img2SeqModel(config)
    model.build()
    model.train(train_set, val_set, lr_schedule)
