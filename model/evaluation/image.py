import numpy as np
import distance
from scipy.misc import imread


from ..utils.general import get_files
from .text import examples


def filter_best_hypo(formulas):
    """Removes extra hypotheses from formulas

    Args:
        formulas: (dict) idx -> string. If 2 consecutive formulas don't have
            consecutive ids, new example

    Returns:
        formulas_filtered: (dict) idx -> string

    """
    formulas_filtered = dict()
    for ex in examples(formulas):
        ref_id, ref_form = ex[0]
        hyp_id, hyp_form = ex[1]
        formulas_filtered[ref_id] = ref_form
        formulas_filtered[hyp_id] = hyp_form

    return formulas_filtered


def image_examples(formulas, mapping):
    """Generator of img path per example

    Args:
        formulas: (dict)
        mapping: list of (path_img, idx). If an exception was raised during the image
            generation, path_img = False
    """
    idx_to_img = {idx: path_img for path_img, idx in mapping}
    for ex in examples(formulas):
        path_images = [idx_to_img[idx] for idx, _ in ex if idx_to_img[idx] is
                       not False]
        if len(path_images) > 1:
            yield path_images


def score_images(paths_gen, dir_images, prepro_img):
    """Returns scores from a dir with images

    Args:
        paths_gen: generator that yields list of paths to image, first one is
            the reference image, second one is the best hypothesis.
        dir_images: (string)
        prepro_img: (lambda function)

    Returns:
        scores: (dict)

    """
    em_tot = l_dist_tot = length_tot = n_ex = 0
    for paths in paths_gen:
        if len(paths) > 1:
            # load images
            img_ref = imread(dir_images + "/" + paths[0])
            img_ref = prepro_img(img_ref)

            img_hyp = imread(dir_images + "/" + paths[1])
            img_hyp = prepro_img(img_hyp)

            # compute scores
            l_dist, length = img_edit_distance(img_ref, img_hyp)
            print paths, l_dist, length
            l_dist_tot += l_dist
            length_tot += length
            if l_dist < 1:
                em_tot += 1
            n_ex += 1

    # compute scores
    scores = dict()
    scores["EM"]  = em_tot / float(n_ex)
    scores["Lev"] = 1. - l_dist_tot / float(length_tot)

    return scores


def img_edit_distance(img1, img2):
    """Computes Levenshtein distance between two images.

    Slices the images into columns and consider one column as a character.

    Args:
        im1, im2: np arrays of shape (H, W, 1)

    Returns:
        column wise levenshtein distance
        max length of the two sequences

    """
    # load the image (H, W)
    img1, img2 = img1[:, :, 0], img2[:, :, 0]

    # transpose and convert to 0 or 1
    img1 = np.transpose(img1)
    h1 = img1.shape[1]
    w1 = img1.shape[0]
    img1 = (img1<=128).astype(np.uint8)

    img2 = np.transpose(img2)
    h2 = img2.shape[1]
    w2 = img2.shape[0]
    img2 = (img2<=128).astype(np.uint8)

    # create binaries for each column
    if h1 == h2:
        seq1 = [''.join([str(i) for i in item]) for item in img1]
        seq2 = [''.join([str(i) for i in item]) for item in img2]
    elif h1 > h2:
        seq1 = [''.join([str(i) for i in item]) for item in img1]
        seq2 = [''.join([str(i) for i in item])+''.join(['0']*(h1-h2)) for
                item in img2]
    else:
        seq1 = [''.join([str(i) for i in item])+''.join(['0']*(h2-h1)) for
                item in img1]
        seq2 = [''.join([str(i) for i in item]) for item in img2]

    # convert each column binary into int
    seq1_int = [int(item,2) for item in seq1]
    seq2_int = [int(item,2) for item in seq2]

    # distance
    l_dist = distance.levenshtein(seq1_int, seq2_int)
    length = float(max(len(seq1_int), len(seq2_int)))

    return l_dist, length
