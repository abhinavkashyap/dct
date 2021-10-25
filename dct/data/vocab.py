from typing import List, Dict, Tuple
import pathlib
from collections import Counter
import itertools
from tqdm import tqdm
import wasabi
from torchtext.vocab import GloVe
import torch


class Vocabulary:
    def __init__(
        self,
        instances: List[List[str]] = None,
        max_vocab_size: int = None,
        add_special_tokens=True,
        glove_name: str = None,
        glove_dim: int = None
    ):
        """

        Parameters
        ----------
        instances: List[List[str]]
            A list of tokenized text.
        max_vocab_size: int
            The maximum vocab size to consider. If None, we consider all the words
        add_special_tokens: bool
            Indicator for adding special tokens
        glove_name: str
            Glove name
        glove_dim: int
            Dimension of the glove vector
        """
        self.instances = instances
        self.max_vocab_size = max_vocab_size
        self.add_special_tokens = add_special_tokens
        self.original_count: int = None
        self.glove_name = glove_name
        self.glove_dim = glove_dim

        if self.instances:
            self.vocab, self.counts = self.make_vocab()
            if self.add_special_tokens:
                self.vocab.extend(["<unk>", "<pad>", "<s>", "</s>"])

        if self.glove_name and self.glove_dim:
            self.glove = GloVe(name=self.glove_name, dim=self.glove_dim)

        self.msg_printer = wasabi.Printer()

    def make_vocab(self) -> Tuple[List[str], List[int]]:
        """Builds the vocab

        Returns
        -------
        List[str]
            A list of words in the vocab

        """
        flat_instances = list(itertools.chain(*self.instances))
        counter = Counter(flat_instances)
        self.original_count = len(counter)
        most_common = list(counter.most_common(self.max_vocab_size))
        most_common_words = [word for word, count in most_common]
        most_common_counts = [count for word, count in most_common]
        return most_common_words, most_common_counts

    def save_vocab(self, filename: pathlib.Path):
        """Saves the vocab with one word per line

        Parameters
        ----------
        filename: pathlib.Path
            The path of the file to store the vocab

        Returns
        -------
        None

        """
        with open(str(filename), "w") as fp:
            for word in tqdm(self.vocab, desc="Saving vocab", total=len(self.vocab)):
                fp.write(word)
                fp.write("\n")

        with open(str(f"{filename}.counts"), "w") as fp:
            for count in tqdm(self.counts, desc="Saving counts", total=len(self.counts)):
                fp.write(str(count))
                fp.write("\n")

    def load_vocab_in(self, filename: pathlib.Path) -> List[str]:
        """Loads the vocab stored in the filename. The filename should contain
        one word per line.

        Parameters
        ----------
        filename: pathlib.Path
            The path of the ifle where the vocab is stored

        Returns
        -------
        List[str]
            A list of strings representing the vocab.
        """
        vocab = []
        with open(str(filename)) as fp:
            for line in tqdm(fp, desc="Loading Vocab"):
                line_ = line.strip().replace("\n", "")
                vocab.append(line_)

        counts = []
        with open(str(f"{filename}.counts")) as fp:
            for line in tqdm(fp, desc="Loading counts"):
                line_ = line.strip().replace("\n", "")
                line_ = int(line_)
                counts.append(line_)

        self.vocab = vocab
        self.counts = counts
        self.msg_printer.good("Finished Loading Vocab")
        return vocab

    def get_idx2word(self) -> Dict[int, str]:
        """A mapping between index and word

        Returns
        -------
        Dict[nt, str]
            A mapping between index and the word

        """
        idx2word = [(idx, word) for idx, word in enumerate(self.vocab)]
        idx2word = dict(idx2word)
        return idx2word

    def get_word2idx(self) -> Dict[str, int]:
        """A mapping between word and index

        Returns
        -------
        Dict[str, int]
            A mapping between word and index.

        """
        word2idx = [(word, idx) for idx, word in enumerate(self.vocab)]
        word2idx = dict(word2idx)
        return word2idx

    def print_statistics(self):
        len_vocab = len(self.vocab)
        top_words = list(zip(self.vocab, self.counts))[:5]

        rows = list()
        rows.append(["Original Vocab", self.original_count])
        rows.append(["Vocab Length", len_vocab])
        rows.append(["Top Words", top_words])

        self.msg_printer.divider("VOCAB STATS")
        formatted = wasabi.table(data=rows, header=["Info", "Stat"], divider=True)
        print(formatted)

    def __len__(self):
        return len(self.vocab)

    def get_glove_embeddings(self):
        """ Return the glove embeddings for the vocab

        Returns
        -------
        torch.Tensor
            shape: |V| x emb_size
            |V| - Size of the vocab
            emb_size: Embedding size

        """
        vocab_size = len(self.vocab)
        idx2word = self.get_idx2word()
        embeddings = []
        for idx in range(vocab_size):
            embedding = self.glove.get_vecs_by_tokens(idx2word[idx], lower_case_backup=True)
            embeddings.append(embedding)

        pretrained_embeddings = torch.stack(embeddings, dim=0)
        return pretrained_embeddings

