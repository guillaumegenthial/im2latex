import os
import numpy as np
import os
import PIL
from PIL import Image
from .general import run


TIMEOUT = 10


def pad_image(img, output_path, pad_size=[8,8,8,8]):
    """
    Pads image with pad size

    Args:
        img: (string) path to image
        output_path: (string) path to output image
    """
    PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = pad_size
    old_im = Image.open(img)
    old_size = (old_im.size[0]+PAD_LEFT+PAD_RIGHT, old_im.size[1]+PAD_TOP+PAD_BOTTOM)
    new_size = old_size
    new_im = Image.new("RGB", new_size, (255,255,255))
    new_im.paste(old_im, (PAD_LEFT,PAD_TOP))
    new_im.save(output_path)


def crop_image(img, output_path):
    """
    Crops image to content

    Args:
        img: (string) path to image
        output_path: (string) path to output image
    """
    old_im = Image.open(img).convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8) # height, width
    nnz_inds = np.where(img_data!=255)
    if len(nnz_inds[0]) == 0:
        old_im.save(output_path)
        return False

    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
    old_im.save(output_path)
    return True


def downsample_image(img, output_path, ratio=2):
    """
    Downsample image by ratio
    """
    assert ratio>=1, ratio
    if ratio == 1:
        return True
    old_im = Image.open(img)
    old_size = old_im.size
    new_size = (int(old_size[0]/ratio), int(old_size[1]/ratio))

    new_im = old_im.resize(new_size, PIL.Image.LANCZOS)
    new_im.save(output_path)
    return True


def convert_to_png(formula, path_out, name, quality=100, density=200):
    """
    Convert latex to png image

    Args:
        formula: (string) of latex
        path_out: (string) path to output directory
        name: (string) name of file
    """
    # write formula into a .tex file
    with open(path_out + "{}.tex".format(name), "w") as f:
        f.write(
    r"""\documentclass[preview]{standalone}
    \begin{document}
        $$ %s $$
    \end{document}""" % (formula))

    # call pdflatex to create pdf
    run("pdflatex -interaction=nonstopmode -output-directory={} {}".format(path_out,
        path_out+"{}.tex".format(name)), TIMEOUT)

    # call magick to convert the pdf into a png file
    run("magick convert -density {} -quality {} {} {}".format(density, quality, path_out+"{}.pdf".format(name),
        path_out+"{}.png".format(name)), TIMEOUT)

    # cropping and downsampling
    img_path = path_out + "{}.png".format(name)
    crop_image(img_path, img_path)
    pad_image(img_path, img_path)
    downsample_image(img_path, img_path)

    # cleaning
    try:
        os.remove(path_out+"{}.aux".format(name))
        os.remove(path_out+"{}.log".format(name))
        os.remove(path_out+"{}.pdf".format(name))
        os.remove(path_out+"{}.tex".format(name))
    except Exception, e:
        print(e)

    return "{}.png".format(name)