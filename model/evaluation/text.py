import os
import sys
import numpy as np
import nltk
import distance


from ..utils.data import load_formulas


def score_files(path_ref, path_hyp):
    """Loads result from file and score it

    Args:
        path_ref: (string) formulas of reference
        path_hyp: (string) formulas of prediction.

    Returns:
        scores: (dict)

    """
    # load formulas
    formulas_ref = load_formulas(path_ref)
    formulas_hyp = load_formulas(path_hyp)

    # tokenize
    refs = [ref.split(' ') for _, ref in formulas_ref.items()]
    hyps = [hyp.split(' ') for _, hyp in formulas_hyp.items()]

    # score
    scores           = dict()
    scores["BLEU-4"] = bleu_score(refs, hyps)
    scores["EM"]     = exact_match_score(refs, hyps)
    scores["Lev"]    = edit_distance(refs, hyps)

    return scores


def exact_match_score(references, hypotheses):
    """Computes exact match scores.

    Args:
        references: list of list of list of ids of tokens (multiple refs)
        hypotheses: list of list of ids of tokens (one hypothesis)

    Returns:
        exact_match: (float) 1 is perfect

    """
    exact_match = 0
    for refs, hypo in zip(references, hypotheses):
        ref = refs[0] # only take first ref
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """Computes bleu score.

    Args:
        references: list of list         (one reference per hypothesis)
        hypotheses: list of list of list (multiple hypotheses)

    Returns:
        BLEU-4 score: (float)

    """
    references = [[ref] for ref in references]
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_4


def edit_distance(references, hypotheses):
    """Computes Levenshtein distance between two sequences.

    Args:
        references: list of lists of list (multiple references per hypothesis)
        hypotheses: list of list (one hypothesis)

    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)

    """
    d_leven, len_tot = 0, 0
    for refs, hypo in zip(references, hypotheses):
        ref = refs[0] # only take first reference
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot
