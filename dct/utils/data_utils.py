import pathlib
import numpy as np
from typing import List
from nltk import ngrams


def ngram_tokenize(text: str, ngram_range: int = 5) -> List[str]:
    """Tokenize a text upto ngram_range.

    Parameters
    ----------
    text: str
        Text to be tokenize
    ngram_range: int
        upto `ngram_range` ngrams are obtained here

    Returns
    -------
    List[str]
        Every string is a ngram extracted from text.

    """
    text = text.split()
    grams = []
    for i in range(1, ngram_range):
        i_grams = [" ".join(gram) for gram in ngrams(text, i)]
        grams.extend(i_grams)
    return grams


# https://gist.github.com/drussellmrichie/47deb429350e2e99ffb3272ab6ab216a
def tree_height(root):
    """
    Find the maximum depth (height) of the dependency parse of a spacy sentence by starting with its root
    Code adapted from https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
    :param root: spacy.tokens.token.Token
    :return: int, maximum height of sentence's dependency parse tree
    """
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)


def get_average_heights(spacy_doc):
    roots = [sent.root for sent in spacy_doc.sents]
    return np.mean([tree_height(root) for root in roots])


def dump_lines(filepath, lines, encoding="utf-8"):
    with open(filepath, "w", encoding=encoding) as f:
        for line in lines:
            line_ = line.strip()
            f.write(line_ + "\n")


def separate_political_transfer_for_dom_clf(
    filename: pathlib.Path, output_dir: pathlib.Path
):
    """Separate the male and female instances from the file for
    domain classification

    Parameters
    ----------
    filename: pathlib.Path
        The filename where the classification examples are stored
        The filename is downloaded from http://tts.speech.cs.cmu.edu/style_models/gender_data.tar
    output_dir: pathlib.Path
        The directory in which the files will be written
    Returns
    -------
    None
    Writes two files republican.clf.male democratic.clf.female

    """
    republican_lines = []
    democratic_lines = []
    with open(str(filename), "r") as fp:
        for line in fp:
            label_text = line.strip().split(" ")
            text = " ".join(label_text[1:])
            if label_text[0].strip() == "republican":
                republican_lines.append(text)
            elif label_text[0].strip() == "democratic":
                democratic_lines.append(text)

    republican_out_file = output_dir.joinpath("republican.clf")
    democratic_out_file = output_dir.joinpath("democratic.clf")
    dump_lines(republican_out_file, lines=republican_lines)
    dump_lines(democratic_out_file, lines=democratic_lines)


def line_length_stats(filename: pathlib.Path):
    """Return the stats of the lengtsh of the sentences.
    We also return the reaw lengths found in the file
    which can be later used for something more useful

        The src file containing one sentence per line
    Returns
    -------
    mean, median, std
    The mean, median and standard deviation
    of the lengths of sentences
    """
    lengths = []
    with open(str(filename)) as fp:
        for line in fp:
            line_ = line.strip()
            words = line_.split()
            num_words = len(words)
            lengths.append(num_words)
    length_mean = np.mean(lengths)
    length_std = np.std(lengths)
    length_median = np.median(lengths)

    return length_mean, length_std, length_median, lengths