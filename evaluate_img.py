from model.configs.config import Config
from model.img2seq import Img2SeqModel
from model.utils.data_generator import DataGenerator
from model.utils.preprocess import greyscale, get_form_prepro
from model.utils.images import build_images
from model.utils.data import load_formulas
from model.evaluation.text import score_files
from model.evaluation.image import score_dirs


if __name__ == "__main__":
    # restore model
    config = Config()
    config.restore_from_dir("results/small/")
    model = Img2SeqModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # load dataset
    test_set = DataGenerator(path_formulas=config.path_formulas_test,
            dir_images=config.dir_images_test, max_iter=config.max_iter,
            path_matching=config.path_matching_test, img_prepro=greyscale,
            form_prepro=get_form_prepro(config.tok_to_id),
            max_len=config.max_length_formula)

    # build images from formulas
    formula_ref = config.dir_formulas_test_result + "ref.txt"
    formula_hyp = config.dir_formulas_test_result + "hyp_0.txt"
    build_images(load_formulas(formula_ref), config.dir_images_test_result_ref)
    build_images(load_formulas(formula_hyp), config.dir_images_test_result_hyp)

    # score the repositories
    scores = score_dirs(config.dir_images_test_result_ref,
            config.dir_images_test_result_hyp, greyscale)
    msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in scores.items()])
    config.logger.info("- Eval Img: {}".format(msg))
