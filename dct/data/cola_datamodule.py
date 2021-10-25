from typing import Optional, List

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class ColaInferDatset(Dataset):
    def __init__(self, lines: List[str], tokenizer):
        super(ColaInferDatset, self).__init__()
        self.lines = lines
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        tokenized = self.tokenizer.encode_plus(
            self.lines[idx], padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ColaDataModule(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer):
        super(ColaDataModule, self).__init__()
        self.tokenizer = tokenizer
        self.cola_dataset = load_dataset("glue", "cola")
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        self.batch_size = hparams.get("batch_size")
        self.num_workers = hparams.get("num_workers")
        self.num_labels = self.cola_dataset["train"].features["label"].num_classes

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self._train_dataset = self.cola_dataset["train"]
            self._train_dataset = self._train_dataset.map(
                lambda e: self.tokenizer(
                    e["sentence"], truncation=True, padding="max_length"
                ),
                batched=True,
            )
            self._train_dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
            )
            self._dev_dataset = self.cola_dataset["validation"]
            self._dev_dataset = self._dev_dataset.map(
                lambda e: self.tokenizer(
                    e["sentence"], truncation=True, padding="max_length"
                ),
                batched=True,
            )
            self._dev_dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
            )

        elif stage == "test":
            self._test_dataset = self.cola_dataset["validation"]
            self._test_dataset = self._test_dataset.map(
                lambda e: self.tokenizer(
                    e["sentence"], truncation=True, padding="max_length"
                ),
                batched=True,
            )
            self._test_dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
            )

        else:
            raise ValueError(f"stage can can be fit or test. You passed {stage}")

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def label2idx(self, prediction: int):
        self.cola_dataset.features["label"].int2str(prediction)

    def idx2label(self, prediction_label: str):
        self.cola_dataset.features["label"].str2int(prediction_label)
