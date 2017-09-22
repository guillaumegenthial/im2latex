from model.utils.data_generator import DataGenerator
from model.img2seq import Img2SeqModel
from model.utils.lr_schedule import LRSchedule
from model.utils.general import Config
from model.utils.text import Vocab
from model.utils.images import greyscale


if __name__ == "__main__":
    # Load configs
    dir_output = "results/small/"
    config = Config(["configs/data_small.json", "configs/vocab_small.json",
                     "configs/training_small.json", "configs/model.json"])
    config.save(dir_output)
    vocab = Vocab(config)

    # Load datasets
    train_set = DataGenerator(path_formulas=config.path_formulas_train,
            dir_images=config.dir_images_train, img_prepro=greyscale,
            max_iter=config.max_iter, bucket=config.bucket_train,
            path_matching=config.path_matching_train,
            max_len=config.max_length_formula,
            form_prepro=vocab.form_prepro)
    val_set = DataGenerator(path_formulas=config.path_formulas_val,
            dir_images=config.dir_images_val, img_prepro=greyscale,
            max_iter=config.max_iter, bucket=config.bucket_val,
            path_matching=config.path_matching_val,
            max_len=config.max_length_formula,
            form_prepro=vocab.form_prepro)

    # Define learning rate schedule
    n_batches_epoch = ((len(train_set) + config.batch_size - 1) //
                        config.batch_size)
    lr_schedule = LRSchedule(lr_init=config.lr_init,
            start_decay=config.start_decay*n_batches_epoch,
            end_decay=config.end_decay*n_batches_epoch,
            end_warm=config.end_warm*n_batches_epoch,
            lr_warm=config.lr_warm,
            lr_min=config.lr_min)

    # Build model and train
    model = Img2SeqModel(config, dir_output, vocab)
    model.build_train(config)
    model.train(config, train_set, val_set, lr_schedule)
