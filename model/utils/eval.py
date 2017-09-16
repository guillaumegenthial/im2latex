import os
import sys
import numpy as np
import nltk
import distance
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from .images import convert_to_png

TIMEOUT = 10


def evaluate(references, hypotheses, rev_vocab, path, id_END):
    """
    Evaluate BLEU and EM scores from txt hypotheses and references
    Write answers in a text file

    Args:
        references: list of lists of list (multiple references per hypothesis)
        hypotheses: list of list of list (multiple hypotheses)
        rev_vocab: (dict) rev_vocab[idx] = word
        path: (string) path where to write results
    """
    hypotheses = truncate_end(hypotheses, id_END)
    write_answers(references, hypotheses, rev_vocab, path)
    scores = dict()
    
    # extract best hypothesis to compute scores
    hypotheses = [hypos[0] for hypos in hypotheses]
    scores["BLEU-4"] = bleu_score(references, hypotheses)
    scores["EM"] = exact_match_score(references, hypotheses)
    return scores
    

def truncate_end(hypotheses, id_END):
    """
    Dummy code to remove the end of each sentence starting from
    the first id_END token.
    """
    trunc_hypotheses = []
    for hypos in hypotheses:
        trunc_hypos = []
        for hypo in hypos:
            trunc_hypo = []
            for id_ in hypo:
                if id_ == id_END:
                    break
                trunc_hypo.append(id_)
            trunc_hypos.append(trunc_hypo)

        trunc_hypotheses.append(trunc_hypos)

    return trunc_hypotheses



def write_answers(references, hypotheses, rev_vocab, path):
    """ 
    Write text answers in file, the format is
        truth
        prediction
        new line
        ...
    """
    assert len(references) == len(hypotheses)

    with open(path, "a") as f:
        for refs, hypos in zip(references, hypotheses):
            ref = refs[0] # only take first ref
            ref = [rev_vocab[idx] for idx in ref]
            f.write(" ".join(ref) + "\n")

            for hypo in hypos:
                hypo = [rev_vocab[idx] for idx in hypo]
                to_write = " ".join(hypo)
                if len(to_write) > 0:
                    f.write(to_write + "\n")

            f.write("\n")


def exact_match_score(references, hypotheses):
    """
    Compute exact match scores.

    Args:
        references: list of list of list of ids of tokens
            (assumes multiple references per exemple). In
            our case we only consider the first reference.

        hypotheses: list of list of ids of tokens
    """
    exact_match = 0
    for refs, hypo in zip(references, hypotheses):
        ref = refs[0] # only take first ref
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """
    Computes bleu score. BLEU-4 has been shown to be the most 
    correlated with human judgement so we use this one.
    """
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_4



def edit_distance(ref, hypo):
    """
    Computes Levenshtein distance between two sequences.

    Args:
        ref, hypo: two lists of tokens

    Returns:
        levenshtein distance
        max length of the two sequences
    """
    d_leven = distance.levenshtein(ref, hypo)
    max_len = float(max(len(ref), len(hypo)))

    return d_leven, max_len


def img_edit_distance(img1, img2):
    """
    Computes Levenshtein distance between two images.
    Slice the images into columns and consider one column as a character.

    Code strongly inspired by Harvard's evaluation scripts.
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
    elif h1 > h2:# pad h2
        seq1 = [''.join([str(i) for i in item]) for item in img1]
        seq2 = [''.join([str(i) for i in item])+''.join(['0']*(h1-h2)) for item in img2]
    else:
        seq1 = [''.join([str(i) for i in item])+''.join(['0']*(h2-h1)) for item in img1]
        seq2 = [''.join([str(i) for i in item]) for item in img2]

    # convert each column binary into int
    seq1_int = [int(item,2) for item in seq1]
    seq2_int = [int(item,2) for item in seq2]

    return edit_distance(seq1_int, seq2_int)


def evaluate_dataset(test_set, dir_plots, prefix=""):
    """
    Render latex formulas into png of reference and hypothesis

    Args:
        test_set: iterable that yields
            list of tuples (np array, formula)
        dir_plots: where to save histograms

    Returns:
        levenhstein distance between formulas
        levenhstein distance between columns of rendered images
    """
    print("Evaluating dataset reconstruction...")
    def _initialize_results():
        return {
            "em_txt": 0,
            "em_img": 0,
            "edit_txt": 0,
            "edit_img": 0,
            "len_txt": 0,
            "len_img": 0,
            "nb_examples": 0,
            "distrib_edit_txt": [],
            "distrib_edit_img": [],
            "distrib_ids": [],
        }

    def _update_results(results, data):
        # increment counts for averages
        results["edit_txt"] += data["edit_txt"]
        results["edit_img"] += data["edit_img"]
        results["len_txt"] += data["len_txt"]
        results["len_img"] += data["len_img"]

        # compute exact matches
        if data["edit_img"] == 0:
            results["em_img"] += 1
        if data["edit_txt"] == 0:
            results["em_txt"] += 1

        # for statistics per image
        norm_edit_img = 1. -  data["edit_img"] / float(data["len_img"])
        norm_edit_txt = 1. -  data["edit_txt"] / float(data["len_txt"])
        results["distrib_edit_txt"].append(norm_edit_txt)
        results["distrib_edit_img"].append(norm_edit_img)
        results["distrib_ids"].append(data["id"])

        # record that we added an example
        results["nb_examples"] += 1



    results, results_best = _initialize_results(), _initialize_results()
    references, hypotheses, hypotheses_best = [], [], []

    # iterate over examples
    for idx, example in enumerate(test_set):
        # print example
        sys.stdout.write("\rAt example {}".format(idx))
        sys.stdout.flush()
        ref = example[0] # ref is the first element
        hypos = example[1:]
        img_ref, formula_ref = ref
        hypo_best = None
        

        # fake hypothesis if we don't have one (failed to compile image)
        if len(hypos) == 0:
            hypos = [(np.zeros((1, 1, 1)), [])]

        # enumerate the different hypotheses
        for idx, hypo in enumerate(hypos):
            img_hypo, formula_hypo = hypo
            edit_img, len_img = img_edit_distance(img_ref, img_hypo)
            edit_txt, len_txt = edit_distance(formula_ref, formula_hypo)
            # the first hypothesis is the one our system would propose
            if idx == 0:
                hypo_best = {"edit_img": edit_img, "edit_txt": edit_txt,
                             "len_img": len_img, "len_txt": len_txt, 
                             "id": idx + 1, "formula": formula_hypo}

                references.append([formula_ref])
                hypotheses.append(formula_hypo)
                _update_results(results, hypo_best)

            # let's look at the other hypothesis (maybe we missed a better one)
            elif edit_img < hypo_best["edit_img"]:
                hypo_best = {"edit_img": edit_img, "edit_txt": edit_txt,
                             "len_img": len_img, "len_txt": len_txt,
                             "id": idx + 1, "formula": formula_hypo}

        # record the best hypothesis
        _update_results(results_best, hypo_best)
        hypotheses_best.append(hypo_best["formula"])
            
    # generate final scores
    scores = dict()

    # scores for the first proposal
    scores["Edit Text"] = 1. - results["edit_txt"] / float(max(results["len_txt"], 1))
    scores["Edit Img"]  = 1. - results["edit_img"] / float(max(results["len_img"], 1))
    scores["EM Text"]   = results["em_txt"] / float(max(results["nb_examples"], 1))
    scores["EM Img"]    = results["em_img"] / float(max(results["nb_examples"], 1))
    scores["BLEU"]      = bleu_score(references, hypotheses)

    # scores for the best proposals
    scores["Edit Text Best"] = 1. - results_best["edit_txt"] / float(max(results_best["len_txt"], 1))
    scores["Edit Img Best"]  = 1. - results_best["edit_img"] / float(max(results_best["len_img"], 1))
    scores["EM Text Best"]   = results_best["em_txt"] / float(max(results_best["nb_examples"], 1))
    scores["EM Img Best"]    = results_best["em_img"] / float(max(results_best["nb_examples"], 1))
    scores["BLEU Best"]      = bleu_score(references, hypotheses_best)

    # plot distributions
    plot_histograms(x0=results["distrib_edit_txt"], x1=results["distrib_edit_img"], 
                    fname=dir_plots + str(prefix) + "_edit_hist")

    plot_histograms(x0=results_best["distrib_edit_txt"], x1=results_best["distrib_edit_img"], 
                    fname=dir_plots + str(prefix) + "_edit_hist_best")

    plot_histogram(x=results_best["distrib_ids"], fname=dir_plots + str(prefix) + "_ids")

    print("\n- done.")
    return scores


def plot_histogram(x, fname, bins=np.arange(0, 7) + 0.5, xlabel="proposal", ylabel="Counts"):
    plt.figure()
    plt.hist(x, bins, histtype='bar', facecolor='green', rwidth=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname + ".png")
    plt.close()


def plot_histograms(x0, x1, fname, bins=np.arange(-0.1, 1.1, 0.1) + 0.05,
                    xlabel0="Edit Txt", xlabel1="Edit Img", ylabel="Counts"):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax0, ax1 = axes.flatten()

    ax0.hist(x0, bins, histtype='bar', facecolor='green', rwidth=0.8)
    ax0.set_xlabel(xlabel0)
    ax0.set_ylabel(ylabel)

    ax1.hist(x1, bins, histtype='bar', facecolor='green', rwidth=0.8)
    ax1.set_xlabel(xlabel1)

    fig.tight_layout()
    plt.savefig(fname + ".png")
    plt.close()


def simple_plots(xs, ys, path_fig):
    for k, v in ys.iteritems():
        plt.figure()
        plt.plot(xs, v)
        plt.xlabel("Max Length")
        plt.ylabel(k)
        plt.savefig("_".join([path_fig] + k.split(" ")) + ".png")
        plt.close()