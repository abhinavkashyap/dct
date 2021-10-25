import pathlib
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from typing import List, Tuple, Dict
from dct.data.vocab import Vocabulary
import numpy as np
from nltk import ngrams
import wasabi
from dct.utils.data_utils import ngram_tokenize


class NgramSaliencyCalcuator:
    def __init__(
        self,
        src_filename: pathlib.Path,
        trg_filename: pathlib.Path,
        vocab_filename: pathlib.Path,
        saliency_threshold: float,
        ngram_range: int = 5,
        probability_smoothing: float = 0.5,
    ):
        """

        Parameters
        ----------
        src_filename: pathlib.Path
            Source corpus filename where every line is tokenized and separated by space.
        trg_filename: pathlib.Path
            Target corpus filename where every line is tokenized and separated by space.
        saliency_threshold: float
            A n-gram is considered as an attribute marker only if the probability of
        ngram_range: int
            Upto `ngram_range` ngrams will be considered to calculate saliency
        probability_smoothing: float
            Smoothing parameter for the n-gram
        """
        self.src_filename = src_filename
        self.trg_filename = trg_filename
        self.vocab_filename = vocab_filename
        self.tokenizer = ngram_tokenize
        self.vocab = Vocabulary()
        self.vocab.load_vocab_in(filename=self.vocab_filename)
        self.saliency_threshold = saliency_threshold
        self.ngram_range = ngram_range
        self.probability_smoothing = probability_smoothing
        self.msg_printer = wasabi.Printer()

        # The count matrix upto self.ngram_range will be calculated using this
        self.vectorizer = CountVectorizer(tokenizer=self.tokenizer)
        self.src_corpus: List[str] = self.mask_corpus_to_unk(self.src_filename)
        self.trg_corpus: List[str] = self.mask_corpus_to_unk(self.trg_filename)

        self.src_count_matrix = self.vectorizer.fit_transform(self.src_corpus)
        self.src_vocab: Dict[str, int] = self.vectorizer.vocabulary_
        self.src_counts = np.sum(self.src_count_matrix, axis=0)
        self.src_counts = np.squeeze(np.asarray(self.src_counts))

        self.trg_count_matrix = self.vectorizer.fit_transform(self.trg_corpus)
        self.trg_vocab: Dict[str, int] = self.vectorizer.vocabulary_
        self.trg_counts = np.sum(self.trg_count_matrix, axis=0)
        self.trg_counts = np.squeeze(np.asarray(self.trg_counts))

    def mask_corpus_to_unk(self, corpus_filename: pathlib.Path) -> List[str]:
        """Replaces the words not in vocab with <unk> and returns all the sentences in the corpus

        Parameters
        ----------
        corpus_filename: pathlib.Path
            The path where the source or target corpus is stored. The file should contain
            one sentence per line and should be tokenized.

        Returns
        -------
        List[str]
            A list of strings with <unk>

        """
        corpus = []
        with open(str(corpus_filename)) as fp:
            for line in tqdm(fp, desc=f"Masking words as <unk> in {corpus_filename}"):
                line_ = line.strip()
                line_ = [
                    word if word in self.vocab.vocab else "<unk>"
                    for word in line_.split()
                ]
                line_ = " ".join(line_)
                corpus.append(line_)
        return corpus

    def _calculate_salience(
        self, corpus: List[str]
    ) -> (List[Tuple[str, float]], List[Tuple[str, float]]):
        """Calculates the source and target saliencies for the corpus

        Parameters
        ----------
        corpus: List[str]
            A corpus of strings

        Returns
        -------
        (List[Tuple[str, float]], List[Tuple[str, float]])
            List of tuple of ngram and the slaiencies

        """
        src_dom_salient_words: List[Tuple[str, float]] = []
        trg_dom_salient_words: List[Tuple[str, float]] = []

        for line in tqdm(corpus, total=len(corpus), desc="Calculating saliency scores"):
            for i in range(1, self.ngram_range):
                i_grams = ngrams(line.split(), i)
                joined = [" ".join(gram) for gram in i_grams]
                for gram in joined:
                    ngram_src_count = (
                        0
                        if gram not in self.src_vocab
                        else self.src_counts[self.src_vocab[gram]]
                    )
                    ngram_trg_count = (
                        0
                        if gram not in self.trg_vocab
                        else self.trg_counts[self.trg_vocab[gram]]
                    )
                    src_salience = (ngram_src_count + self.probability_smoothing) / (
                        ngram_trg_count + self.probability_smoothing
                    )
                    trg_salience = (ngram_trg_count + self.probability_smoothing) / (
                        ngram_src_count + self.probability_smoothing
                    )

                    if src_salience >= self.saliency_threshold:
                        src_dom_salient_words.append((gram, src_salience))
                    elif trg_salience >= self.saliency_threshold:
                        trg_dom_salient_words.append((gram, trg_salience))

        return src_dom_salient_words, trg_dom_salient_words

    def write_saliencies(
        self, src_saliency_filename: pathlib.Path, trg_saliency_filename
    ):
        src_domain_saliencies, _ = self._calculate_salience(self.src_corpus)
        _, trg_domain_saliencies = self._calculate_salience(self.trg_corpus)

        with open(str(src_saliency_filename), "w") as fp:
            for attribute_sal in src_domain_saliencies:
                fp.write(attribute_sal[0])
                fp.write("\n")

        with open(str(trg_saliency_filename), "w") as fp:
            for attribute_sal in trg_domain_saliencies:
                fp.write(attribute_sal[0])
                fp.write("\n")

        self.msg_printer.good(f"Finished writing saliency scores to")


if __name__ == "__main__":
    src_filename = pathlib.Path("/abhinav/dct/data/mcauley_reviews/dvd.transfer.train")
    trg_filename = pathlib.Path("/abhinav/dct/data/mcauley_reviews/electronics.transfer.train")
    vocab_filename = pathlib.Path("/abhinav/dct/data/mcauley_reviews/vocab.txt")
    src_sal_filename = pathlib.Path("/abhinav/dct/data/mcauley_reviews/src_dom.attributes.txt")
    trg_sal_filename = pathlib.Path("/abhinav/dct/data/mcauley_reviews/trg_dom.attributes.txt")

    saliency_calculator = NgramSaliencyCalcuator(
        src_filename=src_filename,
        trg_filename=trg_filename,
        vocab_filename=vocab_filename,
        saliency_threshold=15,
    )
    saliency_calculator.write_saliencies(
        src_saliency_filename=src_sal_filename, trg_saliency_filename=trg_sal_filename
    )
