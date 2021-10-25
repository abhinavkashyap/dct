from dct.console import console
import pathlib
import json
import torch
from dct.models.dann import DANN
from dct.data.dann_dataset import DANNDataset
from dct.data.dann_datamodule import DANNDataModule
from typing import List
import multiprocessing
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from tqdm import tqdm


class DANNInfer:
    def __init__(self, checkpoints_dir: pathlib.Path, hparams_file: pathlib.Path):
        """Recreate the model from the checkpoints directory and the
        hparams file.

        BUGS:
        =====
        The datafiles used during training has to be present in the same place
        during inference as well. If you have moved the data files between
        training and inference this module fails

        TODO:
        =====
        1. Decouple the datamodule and the model.

        Parameters
        ----------
        checkpoints_dir: pathlib.Path
            Checkpoints directory created by pytorch lightning
            to store the model state
        hparams_file: pathlib.Path
            File where the hparams are stored
        """
        self.checkpoints_dir = pathlib.Path(checkpoints_dir)
        self.hparams_file = pathlib.Path(hparams_file)

        if not self.hparams_file.is_file():
            console.print(
                f"[red] {self.hparams_file} does not exist... Loading Model Failed"
            )
            exit(1)

        with open(str(self.hparams_file), "r") as fp:
            self.hparams = json.load(fp)

        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.gpu = self.hparams.get("gpu", -1)
        self.batch_size = self.hparams.get("batch_size", 32)

        self.datamodule = DANNDataModule(self.hparams)
        self.idx2word = self.datamodule.vocab.get_idx2word()
        self.word2idx = self.datamodule.vocab.get_word2idx()
        self.pad_idx = self.word2idx["<pad>"]
        self.sos_idx = self.word2idx["<s>"]
        self.eos_idx = self.word2idx["</s>"]

        self.softmax = nn.Softmax(dim=-1)
        self.model = self._load_model()

    def _load_model(self):
        filenames = list(self.checkpoints_dir.iterdir())
        assert (
            len(filenames) == 1
        ), f"Make sure only the best model is stored in the checkpoints directory"
        best_filename = self.checkpoints_dir.joinpath(filenames[-1])

        with console.status(f"Loading Model"):
            model = DANN.load_from_checkpoint(str(best_filename), hparams=self.hparams)

        if torch.cuda.is_available():
            model.to(self.device)

        console.print("[green] Loaded Model")
        return model

    def predict(self, lines: List[str]):
        dataset = DANNDataset(
            lines=lines,
            labels=None,
            vocabulary=self.datamodule.vocab,
            stage="test",
            max_length=self.hparams.get("max_seq_length"),
            gpu=self.gpu,
        )
        loader = DataLoader(
            dataset,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
        )

        total_batches = np.ceil(len(dataset) // self.batch_size)
        predictions = []
        for batch in tqdm(loader, total=total_batches, desc="ARAE infer"):

            inp_tensor, inp_len_tensor, _, _, _ = batch
            inp_tensor = inp_tensor.to(self.device)
            inp_len_tensor = inp_len_tensor.to(self.device)

            _, trg_taskclf_logits, _, _ = self.model(
                inp_tensor, inp_len_tensor, inp_tensor, inp_len_tensor
            )
            probs = self.softmax(trg_taskclf_logits)
            max, argmax = torch.topk(probs, dim=-1, k=1)
            preds = argmax.squeeze().tolist()
            predictions.extend(preds)

        return predictions


if __name__ == "__main__":
    import wasabi

    experiment_dir = pathlib.Path(
        "/abhinav/dct/experiments/[DANN_DVD_ELECTRONICS]_0.07alpha"
    )
    checkpoints_dir = experiment_dir.joinpath("checkpoints")
    hparams_file = experiment_dir.joinpath("hparams.json")
    inp_file = pathlib.Path("/abhinav/dct/results/dann_transfer/clf/greedy.1.txt.seed1")
    labels_file = (
        "/abhinav/dct/data/mcauley_reviews/electronics.transfer.test.sentiment.txt"
    )
    printer = wasabi.Printer()

    lines = []
    with open(inp_file) as fp:
        for line in fp:
            line_ = line.strip()
            lines.append(line_)

    labels = []
    with open(labels_file) as fp:
        for line in fp:
            line_ = line.strip()
            labels.append(int(line_))

    infer = DANNInfer(checkpoints_dir=checkpoints_dir, hparams_file=hparams_file)
    predictions = infer.predict(lines)
    len_predictions = len(predictions)
    print(predictions)
    num_correct = sum(
        [
            1
            for prediction, correct_label in zip(predictions, labels)
            if prediction == correct_label
        ]
    )
    accuracy = num_correct / len_predictions
    accuracy = np.round(accuracy, 3)
    printer.good(f"accuracy={accuracy * 100}%")
