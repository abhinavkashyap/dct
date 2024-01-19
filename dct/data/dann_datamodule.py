import pytorch_lightning as pl
import pathlib
import wasabi
from loguru import logger
from typing import List, Dict, Optional
from dct.data.data import InputExample
from dct.data.vocab import Vocabulary
from dct.data.dann_dataset import DANNDataset
from dct.data.pl_concat_dataset import ConcatDataset
from torch.utils.data import DataLoader
import torch


class DANNDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(DANNDataModule, self).__init__()
        self.src_dom_train_filename = pathlib.Path(hparams["src_dom_train_filename"])
        self.src_dom_dev_filename = pathlib.Path(hparams["src_dom_dev_filename"])
        self.src_dom_test_filename = pathlib.Path(hparams["src_dom_test_filename"])
        self.trg_dom_train_filename = pathlib.Path(hparams["trg_dom_train_filename"])
        self.trg_dom_dev_filename = pathlib.Path(hparams["trg_dom_train_filename"])
        self.trg_dom_test_filename = pathlib.Path(hparams["trg_dom_test_filename"])
        self.vocab_filename = pathlib.Path(hparams["vocab_file"])
        self.max_seq_length = hparams.get("max_seq_length", 50)
        self.batch_size = hparams.get("batch_size", 32)
        self.debug = hparams.get("debug", False)
        self.loguru_logger = hparams.get(
            "loguru_logger",
        )
        self.msg_printer = wasabi.Printer()
        self.num_processes = hparams.get("num_processes", 1)
        self.droplast = hparams.get("droplast", False)
        self.limit_train_proportion = hparams.get("limit_train_proportion", 1.0)
        self.gpu = hparams.get("gpu", -1)
        self.max_vocab_size = hparams.get("max_vocab_size", None)
        self.glove_name = hparams.get("glove_name")
        self.glove_dim = hparams.get("glove_dim")

        # setup after the vocab is formed
        self.word2idx: Dict[str, int] = None

        # Initialize a new logger and use the defaults
        if self.loguru_logger is None:
            self.loguru_logger = logger

        self.src_train_examples: List[InputExample] = self.read_examples_from_file(
            self.src_dom_train_filename, mode="src-train"
        )
        self.src_dev_examples: List[InputExample] = self.read_examples_from_file(
            self.src_dom_dev_filename, mode="src-dev"
        )
        self.trg_train_examples: List[InputExample] = self.read_examples_from_file(
            self.trg_dom_train_filename, mode="trg-train"
        )
        self.trg_dev_examples: List[InputExample] = self.read_examples_from_file(
            self.trg_dom_dev_filename, mode="trg-dev"
        )

        if self.src_dom_test_filename is not None:
            self.src_test_examples: List[InputExample] = self.read_examples_from_file(
                self.src_dom_test_filename, mode="src-test"
            )
            self.trg_test_examples: List[InputExample] = self.read_examples_from_file(
                self.trg_dom_test_filename, mode="trg-test"
            )

        self.vocab: Vocabulary = Vocabulary()

        self.make_vocab(self.vocab_filename)

        self.src_train_dataset: DANNDataset = None
        self.src_dev_dataset: DANNDataset = None
        self.src_test_dataset: DANNDataset = None
        self.trg_train_dataset: DANNDataset = None
        self.trg_dev_dataset: DANNDataset = None
        self.trg_test_dataset: DANNDataset = None
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        src_train_lines = [" ".join(example.words) for example in self.src_train_examples]
        src_train_labels = [example.label for example in self.src_train_examples]
        src_dev_lines = [" ".join(example.words) for example in self.src_dev_examples]
        src_dev_labels = [example.label for example in self.src_dev_examples]
        src_test_lines = [" ".join(example.words) for example in self.src_test_examples]
        src_test_labels = [example.label for example in self.src_test_examples]

        trg_train_lines = [" ".join(example.words) for example in self.trg_train_examples]
        trg_train_labels = [example.label for example in self.trg_train_examples]
        trg_dev_lines = [" ".join(example.words) for example in self.trg_dev_examples]
        trg_dev_labels = [example.label for example in self.trg_dev_examples]
        trg_test_lines = [" ".join(example.words) for example in self.trg_test_examples]
        trg_test_labels = [example.label for example in self.trg_test_examples]

        if stage == "fit":
            self.src_train_dataset = DANNDataset(
                lines=src_train_lines,
                labels=src_train_labels,
                vocabulary=self.vocab,
                stage="fit",
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu
            )
            self.src_dev_dataset = DANNDataset(
                lines=src_dev_lines,
                labels=src_dev_labels,
                vocabulary=self.vocab,
                stage="fit",
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu
            )

            self.trg_train_dataset = DANNDataset(
                lines=trg_train_lines,
                labels=trg_train_labels,
                vocabulary=self.vocab,
                stage="fit",
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu
            )
            self.trg_dev_dataset = DANNDataset(
                lines=trg_dev_lines,
                labels=trg_dev_labels,
                vocabulary=self.vocab,
                stage="fit",
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu
            )

            self.train_dataset = ConcatDataset([
                self.src_train_dataset, self.trg_train_dataset
            ])

            self.dev_dataset = ConcatDataset([
                self.src_dev_dataset, self.trg_dev_dataset
            ])

        if stage == "test":
            self.src_test_dataset = DANNDataset(
                lines=src_test_lines,
                labels=src_test_labels,
                vocabulary=self.vocab,
                stage="fit",
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu
            )

            self.trg_test_dataset = DANNDataset(
                lines=trg_test_lines,
                labels=trg_test_labels,
                vocabulary=self.vocab,
                stage="fit",
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu
            )
            self.test_dataset = ConcatDataset([
                self.src_test_dataset, self.trg_test_dataset
            ])

    @staticmethod
    def read_examples_from_file(filename: pathlib.Path, mode: str):
        """

        Parameters
        ----------
        filename: pathlib.Path
            The CONLL file where the examples are stored
            The filename should of the format [line1###label\nline2###label]
            Every line should contain a line in one column and label in the second column
            The columns are demarcated by a ### character
            The line should have words separated by space
        mode: str
            It is either of train, dev or test

        Returns
        -------
        List[InputExample]
            Constructs the examples from the file

        """
        examples = []
        with open(str(filename), "r") as fp:
            for idx, line in enumerate(fp):
                line_, label_ = line.split("###")
                line_ = line_.strip()
                # remove the first and the last `""`
                line_ = line_[1:-1]
                label_ = label_.strip()
                # Classification labels are integers
                label_ = int(label_)
                words_ = line_.split()
                example = InputExample(guid=f"{mode}-{idx}", words=words_, label=label_)
                examples.append(example)

        return examples

    def make_vocab(self, vocab_filename):
        self.msg_printer.divider("LOADING VOCAB")
        if not vocab_filename.is_file():
            tokens = []
            src_train_tokens = [example.words for example in self.src_train_examples]
            src_dev_tokens = [example.words for example in self.src_dev_examples]
            trg_train_tokens = [example.words for example in self.trg_train_examples]
            trg_dev_tokens = [example.words for example in self.trg_dev_examples]

            tokens.extend(src_train_tokens)
            tokens.extend(src_dev_tokens)
            tokens.extend(trg_train_tokens)
            tokens.extend(trg_dev_tokens)

            vocab_obj = Vocabulary(instances=tokens, max_vocab_size=self.max_vocab_size,
                                   glove_name=self.glove_name, glove_dim=self.glove_dim)
            vocab_obj.save_vocab(vocab_filename)
            self.vocab = vocab_obj
            self.msg_printer.good(f"Saved vocab in file {vocab_filename}")

        else:
            self.vocab.load_vocab_in(vocab_filename)
            self.msg_printer.good(f"Loaded label vocab from file {vocab_filename}")
            vocab_obj = self.vocab

        self.word2idx = vocab_obj.get_word2idx()
        vocab_obj.print_statistics()
        return vocab_obj

    def get_vocab_size(self):
        return len(self.vocab)

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_processes,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_processes,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_processes,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        return loader
