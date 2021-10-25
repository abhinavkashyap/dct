from typing import List, Dict, Optional
from dct.data.vocab import Vocabulary
from loguru import logger
from torch.utils.data import Dataset
import torch
from dct.utils.tensor_utils import get_seq_seq_tensors


class PLDataset(Dataset):
    def __init__(
        self,
        lines: List[str],
        vocabulary: Vocabulary,
        stage: str = "fit",
        max_length: int = 50,
        loguru_logger=None,
        debug: bool = False,
        gpu: int = -1,
        labels: Optional[List[str]] = None,
        label_vocabulary: Optional[Vocabulary] = None,
    ):
        self.lines = lines
        self.vocabulary = vocabulary
        self.stage = stage
        self.max_length = max_length
        self.tok2idx: Dict[str, int] = self.vocabulary.get_word2idx()
        self.idx2token: Dict[int, str] = self.vocabulary.get_idx2word()
        self.unk_idx: int = self.tok2idx["<unk>"]
        self.start_idx: int = self.tok2idx["<s>"]
        self.loguru_logger = loguru_logger
        self.debug = debug
        self.gpu = gpu
        self.labels = labels
        self.label_vocabulary = label_vocabulary
        self.device = (
            torch.device("cpu") if self.gpu < 0 else torch.device(f"cuda:{self.gpu}")
        )

        if self.loguru_logger is None:
            self.loguru_logger = logger

        if self.labels is not None:
            assert len(self.lines) == len(
                self.labels
            ), f"number of text and labels are not the same: {len(lines)} != {len(labels)}"

            self.labels_len = len(self.labels[0].split())
            self.labels_tok2idx = self.label_vocabulary.get_word2idx()
            self.labels_idx2tok = self.label_vocabulary.get_idx2word()
            self.labels_unkidx = None

            is_label_length_unique = [len(label.split()) for label in self.labels]
            assert (
                len(set(is_label_length_unique)) == 1
            ), f"All the labels should be of the same length"

        assert stage in ["fit", "test"], AssertionError(
            f"stage parameter should be " f"in [fit, test]"
        )
        self.stage = stage

    def __len__(self):
        length_dataset = len(self.lines)
        return length_dataset

    def __getitem__(self, idx):
        line: str = self.lines[idx]

        output_tensors = []
        label_input_tensor = torch.LongTensor([])

        if self.stage == "fit":
            output_tensors = get_seq_seq_tensors(
                input_text=line,
                target_text=line,
                max_length=self.max_length,
                tok2idx=self.tok2idx,
                unk_idx=self.unk_idx,
                debug=self.debug,
            )

        elif self.stage == "test":
            output_tensors = get_seq_seq_tensors(
                input_text=line,
                target_text=None,
                max_length=self.max_length,
                tok2idx=self.tok2idx,
                unk_idx=self.unk_idx,
                debug=self.debug,
            )

        if self.labels is not None:
            label = self.labels[idx]
            tensors = get_seq_seq_tensors(
                input_text=label,
                target_text=None,
                max_length=self.labels_len,
                tok2idx=self.labels_tok2idx,
                unk_idx=self.labels_unkidx,
                debug=self.debug,
                add_special_tokens=False,
            )
            label_input_tensor = tensors[0]

        output_tensors.append(label_input_tensor)

        return output_tensors
