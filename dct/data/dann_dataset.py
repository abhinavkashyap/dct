from typing import List, Dict
from dct.data.vocab import Vocabulary
from loguru import logger
from torch.utils.data import Dataset
import torch
from dct.utils.tensor_utils import get_seq_seq_tensors


class DANNDataset(Dataset):
    def __init__(
        self,
        lines: List[str],
        labels: List[int],
        vocabulary: Vocabulary,
        stage: str = "fit",
        max_length: int = 50,
        loguru_logger=None,
        debug: bool = False,
        gpu: int = -1,
    ):
        super(DANNDataset, self).__init__()
        self.lines = lines
        self.labels = labels
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

        self.device = torch.device("cpu") if self.gpu < 0 else torch.device(f"cuda:{self.gpu}")

        if self.loguru_logger is None:
            self.loguru_logger = logger

        assert stage in ["fit", "test"], AssertionError(
            f"stage parameter should be " f"in [fit, test]"
        )
        self.stage = stage

    def __len__(self):
        length_dataset = len(self.lines)
        return length_dataset

    def __getitem__(self, idx):
        line: str = self.lines[idx]

        if self.stage == "fit":
            output_tensors = get_seq_seq_tensors(
                input_text=line,
                target_text=None,
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
        else:
            raise ValueError(f"stage should be in [fit, test]. You passed {self.stage}")

        if self.labels:
            label: int = self.labels[idx]
            label_tensor = torch.LongTensor([label])
            output_tensors.append(label_tensor)
        return output_tensors
