from collections import defaultdict
import wandb
import pytorch_lightning as pl
from dct.utils.tensor_utils import tensors_to_text
from dct.models.lstm_encoder import LSTMEncoder
from dct.models.gru_encoder import GRUEncoder
import torch.nn as nn
import torch
from dct.models.reverse_layer import ReverseLayerF
import torch.optim as optim
import torchmetrics
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dct.models.dann_linear import DANNLinear
from dct.models.dann_discriminator import DANNDiscriminator


class DANN(pl.LightningModule):
    def __init__(self, hparams):
        super(DANN, self).__init__()
        self.save_hyperparameters(hparams)
        self.pad_idx = hparams.get("pad_idx")
        self.idx2token = hparams.get("idx2token")
        self.enc_embedding_size = hparams.get("enc_emb_size")
        self.enc_hidden_size = hparams.get("enc_hidden_size")
        self.enc_vocab_size = hparams.get("vocab_size")
        self.num_enc_layers = hparams.get("num_enc_layers", 1)
        self.enc_dropout = hparams.get("enc_dropout", 0.0)

        self.domain_disc_out_dim = hparams.get("domain_disc_out_dim")
        self.task_clf_out_dim = hparams.get("task_clf_out_dim")

        self.linear_clf_hidden_dim = hparams.get("linear_clf_hidden_dim")

        self.weight_dom_loss = hparams.get("weight_dom_loss")
        self.no_adv_train = hparams.get("no_adv_train")

        # The \lambda parameter in the original DANN paper
        self.dann_alpha = hparams.get("dann_alpha")
        self.learning_rate = hparams.get("lr")
        self.scheduler_factor = hparams.get("scheduler_factor", 0.1)
        self.scheduler_patience = hparams.get("scheduler_patience", 2)
        self.scheduler_threshold = hparams.get("scheduler_threshold", 0.0001)
        self.scheduler_cooldown = hparams.get("scheduler_cooldown", 0)
        self.scheduler_eps = hparams.get("scheduler_eps", 1e-8)

        self.glove_name = hparams.get("glove_name")
        self.glove_dim = hparams.get("glove_dim")
        self.is_enc_bidir = hparams.get("is_enc_bidir")
        self.domain_disc_in = (
            2 * self.enc_hidden_size if self.is_enc_bidir else self.enc_hidden_size
        )
        self.task_clf_in = (
            2 * self.enc_hidden_size if self.is_enc_bidir else self.enc_hidden_size
        )
        self.use_gru_encoder = hparams.get("use_gru_encoder")

        if self.glove_name and self.glove_dim:
            embedding = hparams.get("glove_embedding")
        else:
            embedding = None

        if self.use_gru_encoder:
            self.feature_extractor = GRUEncoder(
                emsize=self.enc_embedding_size,
                nhidden=self.enc_hidden_size,
                vocab_size=self.enc_vocab_size,
                nlayers=self.num_enc_layers,
                dropout=self.enc_dropout,
                embedding=embedding,
                is_bidirectional=self.is_enc_bidir,
            )

        else:
            self.feature_extractor = LSTMEncoder(
                emsize=self.enc_embedding_size,
                nhidden=self.enc_hidden_size,
                vocab_size=self.enc_vocab_size,
                nlayers=self.num_enc_layers,
                dropout=self.enc_dropout,
                embedding=embedding,
                is_bidirectional=self.is_enc_bidir,
            )

        self.domain_discriminator = DANNDiscriminator(
            in_dim=self.domain_disc_in,
            out_dim=self.domain_disc_out_dim,
            hidden_dim=self.linear_clf_hidden_dim,
        )

        self.task_classifier = DANNLinear(
            in_dim=self.task_clf_in,
            out_dim=self.task_clf_out_dim,
            hidden_dim=self.linear_clf_hidden_dim,
        )

        self.softmax = nn.Softmax(dim=1)
        self.task_loss = nn.CrossEntropyLoss()
        self.dom_disc_loss = nn.CrossEntropyLoss()

        self.src_train_accuracy = torchmetrics.Accuracy()
        self.src_dev_accuracy = torchmetrics.Accuracy()
        self.src_test_accuracy = torchmetrics.Accuracy()
        # The trg domain metrics are only for logging purposes
        # They are not used to make stopping decisions
        self.trg_dev_accuracy = torchmetrics.Accuracy()
        self.trg_test_accuracy = torchmetrics.Accuracy()

    def forward(
        self, src_inp_tensor, src_inp_len_tensor, trg_inp_tensor, trg_inp_len_tensor
    ):
        # B * max_len, hidden_dimension
        src_features = self.feature_extractor(
            src_inp_tensor, src_inp_len_tensor.cpu().view(-1), self.pad_idx
        )
        trg_features = self.feature_extractor(
            trg_inp_tensor, trg_inp_len_tensor.cpu().view(-1), self.pad_idx
        )

        # B * hidden_dimension
        src_features = torch.mean(src_features, dim=1)
        trg_features = torch.mean(trg_features, dim=1)
        # Classifier logits for the source domain

        # B * number_classes
        src_taskclf_logits = self.task_classifier(src_features)
        trg_taskclf_logits = self.task_classifier(trg_features)

        if self.no_adv_train is False:
            src_grl_features = ReverseLayerF.apply(src_features, self.dann_alpha)
            trg_grl_features = ReverseLayerF.apply(trg_features, self.dann_alpha)

            src_domclf_logits = self.domain_discriminator(src_grl_features)
            trg_domclf_logits = self.domain_discriminator(trg_grl_features)
        else:
            src_domclf_logits = None
            trg_domclf_logits = None

        return (
            src_taskclf_logits,
            trg_taskclf_logits,
            src_domclf_logits,
            trg_domclf_logits,
        )

    def training_step(self, batch, batch_idx):
        src_batch, trg_batch = batch
        (src_inp_tensor, src_inp_len_tensor, _, _, _, src_label_tensor) = src_batch
        (trg_inp_tensor, trg_inp_len_tensor, _, _, _, trg_label_tensor) = trg_batch
        bsz = src_inp_tensor.size(0)
        # src_taskclf_logits: B * C  - The logits for the task classifier from source domain
        # trg_taskclf_logits: B * C - The logits for the task classifier from target domain
        # dom_logits: 2B * C - The source and the target domain logits
        # B - Batch size
        # C - Number of classes
        (
            src_taskclf_logits,
            trg_taskclf_logits,
            src_domclf_logits,
            trg_domclf_logits,
        ) = self.forward(
            src_inp_tensor, src_inp_len_tensor, trg_inp_tensor, trg_inp_len_tensor
        )
        task_loss = self.task_loss(src_taskclf_logits, src_label_tensor.view(-1))

        if self.no_adv_train is False:
            src_dom_labels = torch.LongTensor([0] * bsz).to(self.device)
            trg_dom_labels = torch.LongTensor([1] * bsz).to(self.device)
            src_dom_loss = self.dom_disc_loss(
                src_domclf_logits, src_dom_labels.view(-1)
            )
            trg_dom_loss = self.dom_disc_loss(
                trg_domclf_logits, trg_dom_labels.view(-1)
            )
            src_dom_loss = self.weight_dom_loss * src_dom_loss
            trg_dom_loss = self.weight_dom_loss * trg_dom_loss
        else:
            src_dom_loss = torch.FloatTensor([0.0]).to(self.device)
            trg_dom_loss = torch.FloatTensor([0.0]).to(self.device)

        loss = task_loss + src_dom_loss + trg_dom_loss
        src_train_acc = self.src_train_accuracy(
            src_taskclf_logits, src_label_tensor.view(-1)
        )

        self.log("train/src_acc", src_train_acc)
        self.log("train/src_domerr", src_dom_loss)
        self.log("train/trg_domerr", trg_dom_loss)
        self.log("train/task_loss", task_loss)
        self.log("train/loss", loss)

        if batch_idx % 20 == 0:
            src_text = tensors_to_text(tensors=src_inp_tensor, idx2token=self.idx2token)
            src_labels = src_label_tensor.view(-1).tolist()
            self.logger.experiment.log(
                {
                    "Train Sentences": wandb.Table(
                        data=list(zip(src_text, src_labels)), columns=["Text", "Label"]
                    )
                }
            )

        return {
            "loss": loss,
            "src_acc": src_train_acc,
            "src_domerr": src_dom_loss.detach(),
            "trg_domerr": trg_dom_loss.detach(),
            "task_loss": task_loss.detach(),
        }

    def training_epoch_end(self, outputs):
        metrics = list(outputs[0].keys())
        metrics_dict = defaultdict(list)

        for output in outputs:
            for metric in metrics:
                metrics_dict[metric].append(output[metric].cpu().item())

        for metric in metrics:
            self.log(f"train/{metric}", np.mean(metrics_dict[metric]))

    def validation_step(self, batch, batch_idx):
        src_batch, trg_batch = batch
        (src_inp_tensor, src_inp_len_tensor, _, _, _, src_label_tensor) = src_batch
        (trg_inp_tensor, trg_inp_len_tensor, _, _, _, trg_label_tensor) = trg_batch
        bsz = src_inp_tensor.size(0)
        # src_taskclf_logits: B * C  - The logits for the task classifier from source domain
        # trg_taskclf_logits: B * C - The logits for the task classifier from target domain
        # dom_logits: 2B * C - The source and the target domain logits
        # B - Batch size
        # C - Number of classes
        (
            src_taskclf_logits,
            trg_taskclf_logits,
            src_domclf_logits,
            trg_domclf_logits,
        ) = self.forward(
            src_inp_tensor, src_inp_len_tensor, trg_inp_tensor, trg_inp_len_tensor
        )
        task_loss = self.task_loss(src_taskclf_logits, src_label_tensor.view(-1))

        src_dom_labels = torch.LongTensor([0] * bsz).to(self.device)
        trg_dom_labels = torch.LongTensor([1] * bsz).to(self.device)

        if self.no_adv_train is False:
            src_dom_loss = self.dom_disc_loss(
                src_domclf_logits, src_dom_labels.view(-1)
            )
            trg_dom_loss = self.dom_disc_loss(
                trg_domclf_logits, trg_dom_labels.view(-1)
            )
            src_dom_loss = self.weight_dom_loss * src_dom_loss
            trg_dom_loss = self.weight_dom_loss * trg_dom_loss

        else:
            src_dom_loss = torch.FloatTensor([0.0]).to(self.device)
            trg_dom_loss = torch.FloatTensor([0.0]).to(self.device)

        loss = task_loss + src_dom_loss + trg_dom_loss
        src_dev_acc = self.src_dev_accuracy(
            src_taskclf_logits, src_label_tensor.view(-1)
        )
        trg_dev_acc = self.trg_dev_accuracy(
            trg_taskclf_logits, trg_label_tensor.view(-1)
        )

        self.log("dev/src_acc", src_dev_acc)
        self.log("dev/trg_acc", trg_dev_acc)
        self.log("dev/src_domerr", src_dom_loss)
        self.log("dev/trg_domerr", trg_dom_loss)
        self.log("dev/task_loss", task_loss)
        self.log("dev/loss", loss)

        return {
            "loss": loss,
            "src_acc": src_dev_acc,
            "trg_acc": trg_dev_acc,
            "src_domerr": src_dom_loss,
            "trg_domerr": trg_dom_loss,
            "task_loss": task_loss,
        }

    def validation_epoch_end(self, outputs):
        metrics = list(outputs[0].keys())
        metrics_dict = defaultdict(list)

        for output in outputs:
            for metric in metrics:
                metrics_dict[metric].append(output[metric].cpu().item())

        for metric in metrics:
            self.log(f"dev/{metric}", np.mean(metrics_dict[metric]))

    def test_step(self, batch, batch_idx):
        src_batch, trg_batch = batch
        (src_inp_tensor, src_inp_len_tensor, _, _, _, src_label_tensor) = src_batch
        (trg_inp_tensor, trg_inp_len_tensor, _, _, _, trg_label_tensor) = trg_batch
        bsz = src_inp_tensor.size(0)
        # src_taskclf_logits: B * C  - The logits for the task classifier from source domain
        # trg_taskclf_logits: B * C - The logits for the task classifier from target domain
        # dom_logits: 2B * C - The source and the target domain logits
        # B - Batch size
        # C - Number of classes
        (
            src_taskclf_logits,
            trg_taskclf_logits,
            src_domclf_logits,
            trg_domclf_logits,
        ) = self.forward(
            src_inp_tensor, src_inp_len_tensor, trg_inp_tensor, trg_inp_len_tensor
        )
        task_loss = self.task_loss(src_taskclf_logits, src_label_tensor.view(-1))

        src_dom_labels = torch.LongTensor([0] * bsz).to(self.device)
        trg_dom_labels = torch.LongTensor([1] * bsz).to(self.device)

        if self.no_adv_train is False:
            src_dom_loss = self.dom_disc_loss(
                src_domclf_logits, src_dom_labels.view(-1)
            )
            trg_dom_loss = self.dom_disc_loss(
                trg_domclf_logits, trg_dom_labels.view(-1)
            )
            src_dom_loss = self.weight_dom_loss * src_dom_loss
            trg_dom_loss = self.weight_dom_loss * trg_dom_loss
        else:
            src_dom_loss = torch.FloatTensor([0.0]).to(self.device)
            trg_dom_loss = torch.FloatTensor([0.0]).to(self.device)

        loss = task_loss + src_dom_loss + trg_dom_loss
        src_test_acc = self.src_test_accuracy(
            src_taskclf_logits, src_label_tensor.view(-1)
        )
        trg_test_acc = self.trg_test_accuracy(
            trg_taskclf_logits, trg_label_tensor.view(-1)
        )

        self.log("test/src_acc", src_test_acc)
        self.log("test/trg_acc", trg_test_acc)
        self.log("test/src_domerr", src_dom_loss)
        self.log("test/trg_domerr", trg_dom_loss)
        self.log("test/task_loss", task_loss)
        self.log("test/loss", loss)

        return {
            "loss": loss,
            "src_acc": src_test_acc,
            "trg_acc": trg_test_acc,
            "src_domerr": src_dom_loss,
            "trg_domerr": trg_dom_loss,
            "task_loss": task_loss,
        }

    def test_epoch_end(self, outputs):
        metrics = list(outputs[0].keys())
        metrics_dict = defaultdict(list)

        for output in outputs:
            for metric in metrics:
                metrics_dict[metric].append(output[metric].cpu().item())

        for metric in metrics:
            self.log(f"test/{metric}", np.mean(metrics_dict[metric]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            threshold=self.scheduler_threshold,
            threshold_mode="rel",
            cooldown=self.scheduler_cooldown,
            eps=self.scheduler_eps,
            verbose=True,
        )

        return optimizer
