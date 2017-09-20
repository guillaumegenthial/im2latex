def truncate_end(list_of_ids, id_END):
    """Removes the end of the list starting from the first id_END token"""
    list_trunc = []
    for idx in list_of_ids:
        if idx == id_END:
            break
        else:
            list_trunc.append(idx)

    return list_trunc


def write_answers(references, hypotheses, rev_vocab, dir_name, id_END):
    """Writes text answers in files.

    One file for the reference, one file for each hypotheses

    Args:
        references: list of list         (one reference)
        hypotheses: list of list of list (multiple hypotheses)
            hypotheses[0] is a list of all the first hypothesis for all the
            dataset
        rev_vocab: (dict) rev_vocab[idx] = word
        dir_name: (string) path where to write results
        id_END: (int) special id of token that corresponds to the END of
            sentence

    Returns:
        file_names: list of the created files

    """
    def ids_to_str(ids):
        ids = truncate_end(ids, id_END)
        s = [rev_vocab[idx] for idx in ids]
        return " ".join(s)

    def write_file(file_name, list_of_list):
        with open(file_name, "w") as f:
            for l in list_of_list:
                f.write(ids_to_str(l) + "\n")

    file_names = [dir_name + "ref.txt"]
    write_file(dir_name + "ref.txt", references) # one file for the ref
    for i in range(len(hypotheses)):             # one file per hypo
        assert len(references) == len(hypotheses[i])
        write_file(dir_name + "hyp_{}.txt".format(i), hypotheses[i])
        file_names.append(dir_name + "hyp_{}.txt".format(i))

    return file_names
