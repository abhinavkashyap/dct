import json
from typing import Union, List, Optional, Dict
from torch.utils.data import DataLoader
import pathlib
from dct.data.vocab import Vocabulary
from tqdm import tqdm
import math
import wasabi
from dct.data.pl_dataset import PLDataset
from dct.data.pl_concat_dataset import ConcatDataset
import numpy as np
import torch
import pytorch_lightning as pl
from loguru import logger


class DataModule(pl.LightningDataModule):
    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        opts,
    ):
        """
        Parameters
        ----------
        src_train_filename: pathlib.Path
            The src domain/style train filename
        trg_train_filename: pathlib.Path
            The trg domain/style train filename
        src_dev_filename: pathlib.Path
            The src domain/style dev filename
        trg_dev_filename: pathlib.Path
            The trg domain/style dev filename
        src_vocab_filename: pathlib.Path
            The filename where source domain vocab is stored
        trg_vocab_filename: pathlib.Path
            The filename where target domain vocab is stored
        src_test_filename: Optional[pathlib.Path]
            The src domain/style test filename if any. This is the reference dataset
        trg_test_filename: Optional[pathlib.Path}
            The trg domain/style test filename if any. This is the reference dataset.
            This must have gold target style/domain sentences.
        src_train_label_filename: Optional[pathlib.Path]
            Src Train label filename if available
        trg_train_label_filename: Optional[pathlib.Path]
            Trg Train label filename if available
        src_dev_label_filename: Optional[pathlib.Path]
            Src Dev label filename if available
        trg_dev_label_filename: Optional[pathlib.Path]
            Trg Dev label filename if available
        src_test_label_filename: Optional[pathlib.Path]
            Src Test label filename if available
        trg_test_label_filename: Optional[pathlib.Path]
            Trg Test label filename if available
        label_vocab_filename: Optional[pathlib.Path]
            Filename to store the vocab of labels
        max_seq_length: Optional[int]
            All the sequence lengths will be of `max_seq_length`
        batch_size: Optional[int]
            Batch size of examples
        debug: bool
            Whether to print devug information to the console
        loguru_logger: Optional[loguru.Logger]
            A loguru logger to log data pipeline and other information.
            Note that this is different from experiment logging
        limit_train_proportion: float
            Takes only limit_train_proportion proportion of instancse
            from the train dataset for training. You can use this for development puproses
        gpu: int
            The GPU to run the model on
        """
        super(DataModule, self).__init__()
        self.src_train_filename = pathlib.Path(opts["src_train_file"])
        self.trg_train_filename = pathlib.Path(opts["trg_train_file"])
        self.src_dev_filename = pathlib.Path(opts["src_dev_file"])
        self.trg_dev_filename = pathlib.Path(opts["trg_dev_file"])
        self.src_test_filename = pathlib.Path(opts["src_test_file"])
        self.trg_test_filename = pathlib.Path(opts["trg_test_file"])
        self.src_train_label_filename = opts.get("src_train_label_file")
        self.src_dev_label_filename = opts.get("src_dev_label_file")
        self.src_test_label_filename = opts.get("src_test_label_file")
        self.trg_train_label_filename = opts.get("trg_train_label_file")
        self.trg_dev_label_filename = opts.get("trg_dev_label_file")
        self.trg_test_label_filename = opts.get("trg_test_label_file")
        self.vocab_filename = pathlib.Path(opts["vocab_file"])
        self.label_vocab_filename = opts.get("label_vocab_file", None)

        if self.label_vocab_filename is not None:
            self.label_vocab_filename = pathlib.Path(self.label_vocab_filename)
        self.max_seq_length = opts.get("max_seq_length", 50)
        self.batch_size = opts.get("batch_size", 32)
        self.debug = opts.get("debug", False)
        self.loguru_logger = opts.get(
            "loguru_logger",
        )
        self.msg_printer = wasabi.Printer()
        self.num_processes = opts.get("num_processes", 1)
        self.droplast = opts.get("droplast", False)
        self.limit_train_proportion = opts.get("limit_train_proportion", 1.0)
        self.gpu = opts.get("gpu", -1)
        self.max_vocab_size = opts.get("max_vocab_size", None)

        # Initialize a new logger and use the defaults
        if self.loguru_logger is None:
            self.loguru_logger = logger

        self.src_train_lines: List[str] = self._read_instance_file(self.src_train_filename)
        self.trg_train_lines = self._read_instance_file(self.trg_train_filename)
        self.src_dev_lines = self._read_instance_file(self.src_dev_filename)
        self.trg_dev_lines = self._read_instance_file(self.trg_dev_filename)

        if self.src_test_filename is not None:
            self.src_test_lines = self._read_instance_file(self.src_test_filename)
            self.trg_test_lines = self._read_instance_file(self.trg_test_filename)

        self.src_train_label_lines = None
        self.src_dev_label_lines = None
        self.src_test_label_lines = None
        self.trg_train_label_lines = None
        self.trg_dev_label_lines = None
        self.trg_test_label_lines = None
        self.label_vocab = None
        ########################
        # Read labels from file
        ########################
        if self.src_train_label_filename is not None:
            self.src_train_label_lines = self._read_labels_instance_file(
                self.src_train_label_filename
            )
            self.src_dev_label_lines = self._read_labels_instance_file(self.src_dev_label_filename)
            self.src_test_label_lines = self._read_labels_instance_file(
                self.src_test_label_filename
            )

            self.trg_train_label_lines = self._read_labels_instance_file(
                self.trg_train_label_filename
            )
            self.trg_dev_label_lines = self._read_labels_instance_file(self.trg_dev_label_filename)
            self.trg_test_label_lines = self._read_labels_instance_file(
                self.trg_test_label_filename
            )

            self.label_vocab: Vocabulary = Vocabulary()
            self.make_label_vocab(self.label_vocab_filename)

        self.vocab: Vocabulary = Vocabulary()

        self.make_vocab(self.vocab_filename)

        # setup after the vocab is formed
        self.word2idx: Dict[str, int] = None

        # Declaring autoencoder datasets
        self.src_train_autoencoder_dataset: PLDataset = None
        self.trg_train_autoencoder_dataset: PLDataset = None
        self.src_dev_autoencoder_dataset: PLDataset = None
        self.trg_dev_autoencoder_dataset: PLDataset = None
        self.src_test_autoencoder_dataset: PLDataset = None
        self.trg_test_autoencoder_dataset: PLDataset = None

    def prepare_data(self):
        pass

    def _read_instance_file(self, filename: pathlib.Path) -> List[str]:
        lines = []
        with open(str(filename)) as fp:
            for line in tqdm(fp, desc=f"Reading instances from {filename}"):
                line_ = line.strip()
                if line_ != "":
                    lines.append(line_)

        self.loguru_logger.info(f"Read {len(lines)} lines from {filename}")
        return lines

    def _read_labels_instance_file(self, filename: pathlib.Path) -> List[str]:
        instances = []
        with open(str(filename)) as fp:
            for line in tqdm(fp, desc="Reading Labels instances"):
                dict_ = json.loads(line.strip())
                instance = [label_value.strip() for label_name, label_value in dict_.items()]
                instance = " ".join(instance)
                instances.append(instance)

        self.loguru_logger.info(f"Read {len(instances)} lines from {filename}")
        return instances

    def make_vocab(self, vocab_filename):
        self.msg_printer.divider("LOADING VOCAB")
        if not vocab_filename.is_file():
            tokens = []
            src_train_tokens = [instance.split() for instance in self.src_train_lines]
            src_dev_tokens = [instance.split() for instance in self.src_dev_lines]
            trg_train_tokens = [instance.split() for instance in self.trg_train_lines]
            trg_dev_tokens = [instance.split() for instance in self.trg_dev_lines]

            tokens.extend(src_train_tokens)
            tokens.extend(src_dev_tokens)
            tokens.extend(trg_train_tokens)
            tokens.extend(trg_dev_tokens)

            vocab_obj = Vocabulary(instances=tokens, max_vocab_size=self.max_vocab_size)
            vocab_obj.save_vocab(vocab_filename)
            self.vocab = vocab_obj
            self.msg_printer.good(f"Saved label vocab in file {vocab_filename}")

        else:
            self.vocab.load_vocab_in(vocab_filename)
            self.msg_printer.good(f"Loaded label vocab from file {vocab_filename}")
            vocab_obj = self.vocab

        self.word2idx = vocab_obj.get_word2idx()
        vocab_obj.print_statistics()
        return vocab_obj

    def make_label_vocab(self, vocab_filename):
        self.msg_printer.divider("LOADING VOCAB")
        if not vocab_filename.is_file():
            tokens = []
            src_train_tokens = [instance.lower().split() for instance in self.src_train_label_lines]
            src_dev_tokens = [instance.lower().split() for instance in self.src_dev_label_lines]
            trg_train_tokens = [instance.lower().split() for instance in self.trg_train_label_lines]
            trg_dev_tokens = [instance.lower().split() for instance in self.trg_dev_label_lines]

            tokens.extend(src_train_tokens)
            tokens.extend(src_dev_tokens)
            tokens.extend(trg_train_tokens)
            tokens.extend(trg_dev_tokens)

            vocab_obj = Vocabulary(
                instances=tokens,
                max_vocab_size=self.max_vocab_size,
                add_special_tokens=False,
            )
            vocab_obj.save_vocab(vocab_filename)
            self.label_vocab = vocab_obj
            self.msg_printer.good(f"Saved local vocab in file {vocab_filename}")

        else:
            self.label_vocab.load_vocab_in(vocab_filename)
            self.msg_printer.good(f"Loaded vocab from file {vocab_filename}")
            vocab_obj = self.label_vocab

        self.word2idx = vocab_obj.get_word2idx()
        vocab_obj.print_statistics()
        return vocab_obj

    def setup(self, stage: Optional[str] = "fit", randomize_lines: bool = False):
        """

        Parameters
        ----------
        stage: str
            The pipeline of the dataset
        randomize_lines: bool
            If True we randomize the train lines before
            creating the dataset
            Call this method if they are random

        Returns
        -------

        """
        if stage == "fit":
            self.src_train_autoencoder_dataset = PLDataset(
                lines=self.src_train_lines,
                vocabulary=self.vocab,
                stage=stage,
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu,
                labels=self.src_train_label_lines,
                label_vocabulary=self.label_vocab,
            )
            self.trg_train_autoencoder_dataset = PLDataset(
                lines=self.trg_train_lines,
                vocabulary=self.vocab,
                stage=stage,
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu,
                labels=self.trg_train_label_lines,
                label_vocabulary=self.label_vocab,
            )

            self.src_dev_autoencoder_dataset = PLDataset(
                lines=self.src_dev_lines,
                vocabulary=self.vocab,
                stage=stage,
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu,
                labels=self.src_dev_label_lines,
                label_vocabulary=self.label_vocab,
            )
            self.trg_dev_autoencoder_dataset = PLDataset(
                lines=self.trg_dev_lines,
                vocabulary=self.vocab,
                stage=stage,
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu,
                labels=self.trg_dev_label_lines,
                label_vocabulary=self.label_vocab,
            )

        elif stage == "test":
            self.src_test_autoencoder_dataset = PLDataset(
                lines=self.src_test_lines,
                vocabulary=self.vocab,
                stage=stage,
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu,
                labels=self.src_test_label_lines,
                label_vocabulary=self.label_vocab,
            )
            self.trg_test_autoencoder_dataset = PLDataset(
                lines=self.trg_test_lines,
                vocabulary=self.vocab,
                stage=stage,
                max_length=self.max_seq_length,
                loguru_logger=self.loguru_logger,
                debug=self.debug,
                gpu=self.gpu,
                labels=self.trg_test_label_lines,
                label_vocabulary=self.label_vocab,
            )

        self.loguru_logger.info("Finished setting up datasets")

    def train_dataloader(self) -> DataLoader:
        concat_dataset: ConcatDataset = ConcatDataset(
            [
                self.src_train_autoencoder_dataset,
                self.trg_train_autoencoder_dataset,
            ]
        )
        concat_loader: DataLoader = DataLoader(
            dataset=concat_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_processes,
            drop_last=True,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        return concat_loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        src_dev_autoencoder_loader = DataLoader(
            self.src_dev_autoencoder_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_processes,
            drop_last=self.droplast,
            pin_memory=torch.cuda.is_available(),
        )
        trg_dev_autoencoder_loader = DataLoader(
            self.trg_dev_autoencoder_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_processes,
            drop_last=self.droplast,
            pin_memory=torch.cuda.is_available(),
        )
        return [src_dev_autoencoder_loader, trg_dev_autoencoder_loader]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        src_test_autoencoder_loader = DataLoader(
            self.src_test_autoencoder_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_processes,
            drop_last=self.droplast,
        )
        trg_test_autoencoder_loader = DataLoader(
            self.trg_test_autoencoder_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_processes,
            drop_last=self.droplast,
        )

        return [src_test_autoencoder_loader, trg_test_autoencoder_loader]

    def _limit_train_proportion(self):
        """Limits the number of train lines

        Returns
        -------
        None

        """
        if self.limit_train_proportion == 1.0:
            return

        num_src_train_lines = len(self.src_train_lines)
        num_trg_train_lines = len(self.trg_train_lines)

        limited_num_src_train_lines = math.ceil(self.limit_train_proportion * num_src_train_lines)
        limited_num_trg_train_lines = math.ceil(self.limit_train_proportion * num_trg_train_lines)

        self.src_train_lines = self.src_train_lines[: int(limited_num_src_train_lines)]
        self.trg_train_lines = self.trg_train_lines[: int(limited_num_trg_train_lines)]

        self.msg_printer.info(
            f"Originally had {num_src_train_lines} src train lines. You have set limit train proportion to {self.limit_train_proportion}"
            f" Now we have {limited_num_src_train_lines} number of train lines"
        )
        self.msg_printer.info(
            f"Originally had {num_trg_train_lines} trg train lines. You have set limit train proportion to {self.limit_train_proportion}"
            f" Now we have {limited_num_trg_train_lines} number of trg train lines"
        )

    def sample_src_train_random(self):
        return self._sample_random(self.src_train_autoencoder_dataset)

    def sample_trg_train_random(self):
        return self._sample_random(self.trg_train_autoencoder_dataset)

    def _sample_random(self, dataset: PLDataset):
        length = len(dataset)
        idxs = np.random.choice(range(length), replace=False, size=self.batch_size)
        batch_ = []
        for index in idxs:
            tensors = dataset[index]
            batch_.append(tensors)
        batch_ = list(zip(*batch_))
        final_batch = []
        for tensors in batch_:
            final_tensor = torch.stack(tensors, dim=0)
            final_batch.append(final_tensor)

        return final_batch

    def get_vocab_size(self):
        return len(self.vocab)

    def get_label_vocab_size(self):
        return len(self.label_vocab)
