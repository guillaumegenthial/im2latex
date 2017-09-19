from model.utils.data_generator import DataGenerator
from model.img2seq import Img2SeqModel
from model.configs.config import Config
from model.configs.small import Small
from model.utils.preprocess import greyscale, get_form_prepro


if __name__ == "__main__":
    # restore model
    config = Small()
    model = Img2SeqModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # evaluate
    test_set = DataGenerator(path_formulas=config.path_formulas_test,
            dir_images=config.dir_images_test, max_iter=config.max_iter,
            path_matching=config.path_matching_test, img_prepro=greyscale,
            form_prepro=get_form_prepro(config.tok_to_id),
            max_len=config.max_length_formula)

    model.evaluate(test_set,
            params={"path_formulas_result": config.path_formulas_test_result})