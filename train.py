from utils.dataset import Dataset
from models.model import Model
from configs.config import Config, Test
from utils.preprocess import greyscale, get_form_prepro, compose
from utils.data_utils import minibatches, pad_batch_formulas, \
    pad_batch_images
from utils.lr_schedule import LRSchedule


if __name__ == "__main__":
    # Load config
    # config = Config()
    config = Test() # for test purposes

    # Load datasets
    train_set =  Dataset(path_formulas=config.path_formulas, dir_images=config.dir_images,
                    path_matching=config.path_matching_train, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(config.vocab), max_len=config.max_length_formula,
                    max_iter=config.max_iter)

    val_set   =  Dataset(path_formulas=config.path_formulas, dir_images=config.dir_images,
                    path_matching=config.path_matching_val, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(config.vocab), max_len=config.max_length_formula,
                    max_iter=config.max_iter)

    test_set  =  Dataset(path_formulas=config.path_formulas, dir_images=config.dir_images,
                    path_matching=config.path_matching_test, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(config.vocab), max_len=config.max_length_formula,
                    max_iter=config.max_iter)

    test_set = train_set = val_set # for test purposes

    n_batches_epoch = (len(train_set) + config.batch_size - 1) // config.batch_size
    
    lr_schedule = LRSchedule(lr_init=config.lr_init, lr_min=config.lr_min, 
                            start_decay=config.start_decay*n_batches_epoch,
                            end_decay=config.end_decay*n_batches_epoch,
                            lr_warm=config.lr_warm,
                            end_warm=config.end_warm*n_batches_epoch)

    # Build model
    model = Model(config)
    model.build()
    model.train(train_set, val_set, lr_schedule)
    model.evaluate(test_set, config.model_output, config.path_results_final)