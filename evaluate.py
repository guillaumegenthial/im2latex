from model.configs.config import Config
from model.configs.small import Small
from model.img2seq import Img2SeqModel
from model.utils.data_generator import DataGenerator
from model.utils.preprocess import greyscale, get_form_prepro
from model.utils.images import build_images
from model.utils.data import load_formulas
from model.evaluation.text import score_files
from model.evaluation.image import score_dirs


if __name__ == "__main__":
    # restore model and load dataset
    config = Small()
    model = Img2SeqModel(config)
    model.build()
    model.restore_session(config.dir_model)

    test_set = DataGenerator(path_formulas=config.path_formulas_test,
            dir_images=config.dir_images_test, max_iter=config.max_iter,
            path_matching=config.path_matching_test, img_prepro=greyscale,
            form_prepro=get_form_prepro(config.tok_to_id),
            max_len=config.max_length_formula)

    # text
    files, perplexity = model.write_prediction(test_set,
            params={"dir_name": config.dir_formulas_test_result})
    formula_ref, formula_hyp = files[0], files[1]
    scores = score_files(formula_ref, formula_hyp)
    scores["perplexity"] = perplexity
    msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in scores.items()])
    config.logger.info("- Eval Txt: {}".format(msg))

    # image
    build_images(load_formulas(formula_ref), config.dir_images_test_result_ref)
    build_images(load_formulas(formula_hyp), config.dir_images_test_result_hyp)

    scores = score_dirs(config.dir_images_test_result_ref,
            config.dir_images_test_result_hyp, greyscale)
    msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in scores.items()])
    config.logger.info("- Eval Img: {}".format(msg))

