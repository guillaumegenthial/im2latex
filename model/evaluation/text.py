import os
import sys
import numpy as np
import nltk
import distance


from ..utils.data import load_formulas


"""
NOTE: The file containing the answer has to follow the following format

    truth1      (example 1)
    pred1.1
    pred1.2
                (new line = new example)
    truth2      (example 2)
    pred2.1
    ...
"""


def truncate_end(list_of_ids, id_END):
    """Removes the end of the list starting from the first id_END token"""
    list_trunc = []
    for idx in list_of_ids:
        if idx == id_END:
            break
        else:
            list_trunc.append(idx)

    return list_trunc


def write_answers(references, hypotheses, rev_vocab, path, id_END):
    """Writes text answers in file

    Args:
        references: list of list         (one reference)
        hypotheses: list of list of list (multiple hypotheses)
        rev_vocab: (dict) rev_vocab[idx] = word
        path: (string) path where to write results
        id_END: (int) special id of token that corresponds to the END of
            sentence

    """
    assert len(references) == len(hypotheses)

    def ids_to_str(ids):
        ids = truncate_end(ids, id_END)
        s = [rev_vocab[idx] for idx in ids]
        return " ".join(s)

    with open(path, "w") as f:
        for ref, hypos in zip(references, hypotheses):
            # write reference
            f.write(ids_to_str(ref) + "\n")

            # write hypotheses
            hypos = [ids_to_str(hypo) for hypo in hypos]
            hypos = [hypo + "\n" for hypo in hypos if len(hypo) > 0]
            f.write("".join(hypo for hypo in hypos))

            # new example
            f.write("\n")


def unzip_formulas(formulas):
    """Build reference and hypotheses from formulas

    Args:
        formulas: (dict) idx -> string. If 2 consecutive formulas don't have
            consecutive ids, new example

    Returns:
        reference: list of list          (one reference)
        hypotheses: list of list of list (multiple hypothesis)

    """
    last_idx = -1
    references, hypotheses, ex = [], [], []
    for idx, form in formulas.items():
        if idx - last_idx > 1: # new example
            if len(ex) > 1:
                references.append(ex[0])
                hypotheses.append(ex[1:])
            ex = []

        ex.append(form.split(' '))
        last_idx = idx

    return references, hypotheses


def score_file(path):
    """Loads result from file and score it

    Args:
        path: (string) path with results

    Returns:
        scores: (dict)

    """
    formulas = load_formulas(path)
    references, hypotheses = unzip_formulas(formulas)

    # only take the first hypo for each example (best one)
    best_hypotheses = [hypos[0] for hypos in hypotheses]
    scores           = dict()
    scores["BLEU-4"] = bleu_score(references, best_hypotheses)
    scores["EM"]     = exact_match_score(references, best_hypotheses)
    scores["Lev"]    = edit_distance(references, best_hypotheses)

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

    return 1 - d_leven / len_tot
