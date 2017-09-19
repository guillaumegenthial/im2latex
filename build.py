from model.utils.data_generator import DataGenerator
from model.utils.data import build_vocab, write_vocab
from model.configs.config import Config
from model.utils.images import build_images

if __name__ == "__main__":
    config = Config()

    # datasets
    train_set = DataGenerator(
        path_formulas=config.path_formulas_train,
        dir_images=config.dir_images_train,
        path_matching=config.path_matching_train)
    test_set  = DataGenerator(
        path_formulas=config.path_formulas_test,
        dir_images=config.dir_images_test,
        path_matching=config.path_matching_test)
    val_set   = DataGenerator(
        path_formulas=config.path_formulas_val,
        dir_images=config.dir_images_val,
        path_matching=config.path_matching_val)

    # produce images and matching files
    buckets = [
        (240, 100), (320, 80), (400, 80), (400, 100), (480, 80), (480, 100),
        (560, 80), (560, 100), (640, 80), (640, 100), (720, 80), (720, 100),
        (720, 120), (720, 200), (800, 100), (800, 320), (1000, 200),
        (1000, 400), (1200, 200), (1600, 200), (1600, 1600)
        ]

    val_set.build(buckets=buckets)

    # vocab
    vocab = build_vocab([train_set], min_count=config.min_count_tok)
    write_vocab(vocab, config.path_vocab)
