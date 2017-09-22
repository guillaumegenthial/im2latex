from scipy.misc import imread
import PIL
from PIL import Image


from model.utils.data_generator import DataGenerator
from model.img2seq import Img2SeqModel
from model.utils.general import Config
from model.utils.text import Vocab
from model.utils.images import greyscale

from model.utils.data import load_formulas
from model.evaluation.text import score_files
from model.evaluation.image import score_dirs


def interactive_shell(model):
    """Creates interactive shell to play with model
    Args:
        model: instance of NERModel
    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
Enter a path to a file
input> data/images_test/0.png""")

    while True:
        try:
            # for python 2
            img_path = raw_input("input> ")
        except NameError:
            # for python 3
            img_path = input("input> ")

        if img_path == "exit":
            break


        old_im = Image.open(img_path)
        old_size = old_im.size
        target = 280
        ratio = old_size[0] / target
        # new_size = (target, int(old_size[1]/ratio))
        new_size = (280, 40)
        new_im = old_im.resize(new_size, PIL.Image.LANCZOS)
        new_im.save("temp.png")

        img = imread("temp.png")
        img = greyscale(img)
        hyps = model.predict(img)

        model.logger.info(hyps[0])


if __name__ == "__main__":
    # restore config and model
    dir_output = "results/google/under_50_beam_5_cnn_positional/"
    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")
    vocab = Vocab(config_vocab)

    model = Img2SeqModel(config_model, dir_output, vocab)
    model.build_pred()
    model.restore_session(dir_output + "model.weights/")

    interactive_shell(model)
