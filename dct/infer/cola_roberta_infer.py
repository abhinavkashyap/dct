import pathlib
from dct.models.roberta_finetune import RobertaFineTune
from dct.data.cola_datamodule import ColaDataModule, ColaInferDatset
from transformers import AutoTokenizer
import wasabi
import json
from typing import List
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class ColaRobertaInfer:
    def __init__(self, checkpoints_dir: pathlib.Path, hparams_file: pathlib.Path):
        self.checkpoints_dir = pathlib.Path(checkpoints_dir)
        self.hparams_file = pathlib.Path(hparams_file)

        with open(str(self.hparams_file), "r") as fp:
            self.hparams = json.load(fp)

        self.encoder_name = self.hparams["encoder_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.batch_size = 64
        self.num_workers = 16
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.msg_printer = wasabi.Printer()

        self.datamodule = ColaDataModule(self.hparams, tokenizer=self.tokenizer)
        self.model = self._load_model()

    def _load_model(self):
        filenames = list(self.checkpoints_dir.iterdir())
        assert (
            len(filenames) == 1
        ), f"Make sure only the best model is stored in the checkpoints directory"
        best_filename = self.checkpoints_dir.joinpath(filenames[-1])

        with self.msg_printer.loading(f"Loading Model"):
            model = RobertaFineTune.load_from_checkpoint(
                str(best_filename),
                datamodule=self.datamodule,
            )
            if torch.cuda.is_available():
                model.to(self.device)

        self.msg_printer.good(f"Finished Loading Model from {best_filename}")
        return model

    def predict(self, sentences: List[str]) -> torch.Tensor:

        dataset = ColaInferDatset(lines=sentences, tokenizer=self.tokenizer)
        loader = DataLoader(
            dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
        )
        total_batches = np.ceil(len(dataset) // self.batch_size)

        all_preds = []
        for batch in tqdm(loader, total=total_batches, desc="Cola infer", leave=False):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            batch_ = {"input_ids": input_ids, "attention_mask": attention_mask}
            model_outputs = self.model(batch_)
            logits = model_outputs.logits
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.tolist())

        return torch.LongTensor(all_preds)


if __name__ == "__main__":
    experiment_dir = pathlib.Path(
        "/home/rkashyap/abhi/synarae/experiments/debug_cola_roberta-base_25e/checkpoints"
    )
    params_file = pathlib.Path(
        "/home/rkashyap/abhi/synarae/experiments/debug_cola_roberta-base_25e/hparams.json"
    )
    infer = ColaRobertaInfer(experiment_dir, params_file)
    predictions = infer.predict(["This is a good sentence", "the the the"])
    print(predictions)
