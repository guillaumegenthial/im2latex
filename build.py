from model.utils.data_generator import DataGenerator
from model.utils.text import build_vocab, write_vocab
from model.utils.image import build_images
from model.utils.general import Config


if __name__ == "__main__":
    data_config = Config("configs/small_data.json")

    # datasets
    train_set = DataGenerator(
        path_formulas=data_config.path_formulas_train,
        dir_images=data_config.dir_images_train,
        path_matching=data_config.path_matching_train)
    test_set  = DataGenerator(
        path_formulas=data_config.path_formulas_test,
        dir_images=data_config.dir_images_test,
        path_matching=data_config.path_matching_test)
    val_set   = DataGenerator(
        path_formulas=data_config.path_formulas_val,
        dir_images=data_config.dir_images_val,
        path_matching=data_config.path_matching_val)

    # produce images and matching files
    train_set.build(buckets=data_config.buckets)
    test_set.build(buckets=data_config.buckets)
    val_set.build(buckets=data_config.buckets)

    # vocab
    vocab_config = Config("configs/small_vocab.json")
    vocab = build_vocab([train_set], min_count=vocab_config.min_count_tok)
    write_vocab(vocab, vocab_config.path_vocab)
