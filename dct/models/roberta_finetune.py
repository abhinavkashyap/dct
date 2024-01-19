# -*- coding: utf-8 -*-
import logging as log
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from typing import Dict, Any, List

from transformers import RobertaForSequenceClassification


class RobertaFineTune(pl.LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentences.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams: Dict[str, Any], datamodule) -> None:
        super(RobertaFineTune, self).__init__()
        self.save_hyperparameters(hparams)
        self.batch_size = hparams["batch_size"]

        # Build Data module
        self.dm = datamodule

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if hparams["nr_frozen_epochs"] > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams["nr_frozen_epochs"]

        self.train_acc = Accuracy()
        self.dev_acc = Accuracy()
        self.test_acc = Accuracy()

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        self.transformer_model = RobertaForSequenceClassification.from_pretrained(
            self.hparams["encoder_model"], output_hidden_states=True
        )

        # set the number of features our encoder model will return...
        if self.hparams["encoder_model"] == "google/bert_uncased_L-2_H-128_A-2":
            self.encoder_features = 128
        elif self.hparams["encoder_model"] == "roberta-large":
            self.encoder_features = 1024
        else:
            self.encoder_features = 768

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.dm.num_labels),
        )

    def __build_loss(self):
        """ Initializes the loss function/s. """
        self._loss = nn.CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.transformer_model.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.transformer_model.parameters():
            param.requires_grad = False
        self._frozen = True

    def forward(self, inputs_dict):
        """

        Parameters
        ----------
        inputs_dict: Dict[str, Any]
        A dictionary returned by the datamodule.
        Look at cola_datamodule.py to understand the dictionaries returned
        {
        "input_ids": []
        "token_type_ids": [],
        "attention_mask": [],
        "label": []
        }
        All of it will be passed to the model

        Returns
        -------
        model_outputs: Tuple
        loss: torch.FloatTensor
        logits: torch.FloatTensor
        hidden_states= tuple(torch.)

        """
        # Run BERT model.

        # remove the label from the input dict if available
        if "label" in inputs_dict:
            labels = inputs_dict.pop("label")
        else:
            labels = None

        model_outputs = self.transformer_model(**inputs_dict, labels=labels)

        return model_outputs

    def training_step(self, batch, batch_nb: int, *args, **kwargs) -> dict:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        labels = batch["label"]
        model_out = self.forward(batch)

        logits = model_out.logits
        loss = model_out.loss

        _, preds = torch.max(logits, dim=1)
        train_acc = self.train_acc(preds, labels)

        self.log(
            name="train_acc",
            value=train_acc,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            name="train/loss",
            value=loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb, *args, **kwargs) -> dict:
        """Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        labels = batch["label"]
        model_out = self.forward(batch)

        logits = model_out.logits
        loss = model_out.loss

        _, preds = torch.max(logits, dim=1)
        dev_acc = self.train_acc(preds, labels)

        # can also return just a scalar instead of a dict (return loss_val)
        return {"loss": loss.item(), "dev_acc": dev_acc.item()}

    def validation_epoch_end(self, outputs: List[Any]):
        dev_accs = []
        dev_losses = []
        for output in outputs:
            dev_acc_ = output["dev_acc"]
            dev_loss_ = output["loss"]
            dev_accs.append(dev_acc_)
            dev_losses.append(dev_loss_)

        mean_dev_acc = np.mean(dev_accs)
        mean_dev_loss = np.mean(dev_losses)

        self.log(name="dev_acc", value=mean_dev_acc, logger=True, prog_bar=True)
        self.log(name="dev_loss", value=mean_dev_loss, logger=True, prog_bar=True)

    def test_step(self, batch, batch_nb, *args, **kwargs) -> dict:
        """Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        labels = batch["label"]
        model_out = self.forward(batch)

        logits = model_out.logits

        _, preds = torch.max(logits, dim=1)
        test_acc = self.train_acc(preds, labels)

        return {"test_acc": test_acc.item()}

    def test_epoch_end(self, outputs: List[Any]):
        test_accs = []
        for output in outputs:
            test_acc_ = output["test_acc"]
            test_accs.append(test_acc_)

        mean_test_acc = np.mean(test_accs)

        self.log(name="test_acc", value=mean_test_acc, logger=True, prog_bar=True)

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.transformer_model.parameters(),
                "lr": self.hparams["encoder_lr"],
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams["clf_lr"])
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()
