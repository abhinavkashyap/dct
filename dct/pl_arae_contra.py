import pytorch_lightning as pl
import wandb
from dct.models.autoencoder import Seq2Seq
from dct.models.discriminator import Critic
from dct.data.pl_datamodule import DataModule
from typing import Dict, Any, List, Optional, Callable
import torch.optim as optim
import wasabi
import pathlib
import torch
from dct.utils.tensor_utils import tensors_to_text
import numpy as np
import torch.nn as nn
from torch.optim import Optimizer
from pytorch_lightning.metrics import Accuracy
from torchmetrics import F1
from dct.evaluation.sim.SIM import SIM
from dct.infer.cola_roberta_infer import ColaRobertaInfer
import itertools
from dct.models.contra_loss import ContraLoss
from dct.utils.pairwise_dist import pairwise_attr_intersection
from dct.utils.generate_constraints import GenerateLabels
from dct.models.sup_contra_loss import SupConLoss
from rich.traceback import install

install(show_locals=False)


class PLARAE(pl.LightningModule):
    def __init__(
        self,
        src_autoencoder: Seq2Seq,
        trg_autoencoder: Seq2Seq,
        trg_discriminator: Critic,
        datamodule: DataModule,
        ft_model,
        hparams: Dict[str, Any],
        cola_roberta_checkpoints_dir: Optional[pathlib.Path] = None,
        cola_roberta_json_file: Optional[pathlib.Path] = None,
        contrastive_baseline_classifier: Optional[Critic] = None,
        src_saliency_file: pathlib.Path = None,
        trg_saliency_file: pathlib.Path = None,
    ):
        """

        Parameters
        ----------
        src_autoencoder
        trg_autoencoder
        trg_discriminator
        datamodule
        ft_model
        hparams
        cola_roberta_checkpoints_dir
        cola_roberta_json_file
        contrastive_baseline_classifier: Optional[Critic]
            Use this for the baseline setup where a classifier is used to differentiate
            between the different linguistic attributes instead of contrastive loss
        src_saliency_file
        trg_saliency_file
        """
        super(PLARAE, self).__init__()
        self.save_hyperparameters(hparams)
        self.temperature = 1
        self.src_autoencoder = src_autoencoder
        self.trg_autoencoder = trg_autoencoder
        self.trg_discriminator = trg_discriminator
        self.dm = datamodule
        self.ft_model = ft_model
        self.dm.prepare_data()
        self.word2idx = self.dm.vocab.get_word2idx()
        self.idx2word = self.dm.vocab.get_idx2word()
        self.labelword2idx = self.dm.label_vocab.get_word2idx()
        self.labelidx2word = self.dm.label_vocab.get_idx2word()
        self.pad_idx = self.word2idx["<pad>"]
        self.sos_idx = self.word2idx["<s>"]
        self.eos_idx = self.word2idx["</s>"]
        self.criterion_src_ae = self._build_cross_entropy_loss(pad_idx=self.pad_idx)
        self.criterion_trg_ae = self._build_cross_entropy_loss(pad_idx=self.pad_idx)
        self.cola_roberta_checkpoints_dir = cola_roberta_checkpoints_dir
        self.cola_roberta_json_file = cola_roberta_json_file
        self.contrastive_baseline_classifier = contrastive_baseline_classifier
        self.src_saliency_file = src_saliency_file
        self.trg_saliency_file = trg_saliency_file
        self.sim_model_file = hparams["sim_model"]
        self.sim_sentencepiece_model_file = hparams["sim_sentencepiece_model"]
        self.semantic_sim = SIM(
            self.sim_model_file, self.sim_sentencepiece_model_file, run_on_cpu=True
        )

        self.exp_name = hparams["exp_name"]
        self.exp_dir = (
            pathlib.Path(hparams["exp_dir"])
            if isinstance(hparams["exp_dir"], str)
            else hparams["exp_dir"]
        )
        self.n_epochs = hparams["n_epochs"]
        self.src_autoencoder = src_autoencoder
        self.trg_autoencoder = trg_autoencoder
        self.trg_discriminator = trg_discriminator
        self.ae_anneal_noise_every = hparams["ae_anneal_noise_every"]
        self.ae_anneal_noise = hparams["ae_anneal_noise"]
        self.gpu = hparams["gpu"]
        self.grad_clip_norm = hparams["grad_clip_norm"]
        self.lr_ae = hparams["lr_ae"]
        self.lr_disc = hparams["lr_disc"]
        self.lr_gen = hparams["lr_gen"]
        self.batch_size = hparams["batch_size"]
        self.log_ae_every_iter = hparams["log_ae_every_iter"]
        self.max_seq_length = hparams["max_seq_length"]
        self.gan_gp_lambda = hparams["gan_gp_lambda"]
        self.grad_lambda = hparams["grad_lambda"]
        self.autoencoder_training_off = hparams["autoencoder_training_off"]
        self.disc_training_off = hparams["disc_training_off"]
        self.adv_training_off = hparams["adv_training_off"]
        self.disc_num_steps = hparams["disc_num_steps"]
        self.wasabi_print = wasabi.Printer()
        self.generated_sents_dir = self.exp_dir.joinpath("generated_sents")
        self.generate_through_trg = hparams["generate_through_trg"]
        self.single_encoder_two_decoders = hparams["single_encoder_two_decoders"]
        self.ft_style_mapping = {"__label__1": 0, "__label__2": 1}
        self.lambda_ae = hparams["lambda_ae"]
        self.lambda_cc = hparams["lambda_cc"]
        self.lambda_adv = hparams["lambda_adv"]
        self.num_nearest_labels = hparams["num_nearest_labels"]
        self.discriminator_contrastive_learning_off = hparams[
            "discriminator_contrastive_learning_off"
        ]
        self.encoder_contrastive_learning_off = hparams["encoder_contrastive_learning_off"]
        self.normalize_contrastive_tensors = hparams["normalize_contrastive_tensors"]
        self.contrastive_loss_reduction = hparams.get("contrastive_loss_reduction", "sum")
        self.use_official_sup_con_loss = hparams.get("use_official_sup_con_loss")
        self.weight_contrastive_loss = hparams.get("weight_contrastive_loss", 1.0)
        self.generator_only_contrastive_learning = hparams.get(
            "generator_only_contrastive_learning"
        )
        self.use_disc_attr_clf = hparams.get("use_disc_attr_clf", False)
        self.use_enc_attr_clf = hparams.get("use_enc_attr_clf", False)

        self.contra_loss = ContraLoss(
            temperature=self.temperature, reduction=self.contrastive_loss_reduction
        )
        self.sup_con_loss = SupConLoss()
        self.dev_transfer_accuracy = Accuracy()
        self.test_transfer_accuracy = Accuracy()
        self.dev_len_maintained_acc = Accuracy()
        self.dev_len_maintained_f1 = F1()
        self.test_len_maintained_acc = Accuracy()
        self.test_len_maintained_f1 = F1()
        self.dev_pron_maintained_acc = Accuracy()
        self.dev_pron_maintained_f1 = F1()
        self.test_pron_maintained_acc = Accuracy()
        self.test_pron_maintained_f1 = F1()
        self.dev_num_adjectives_maintained_acc = Accuracy()
        self.dev_num_adjectives_maintained_f1 = F1()
        self.test_num_adjectives_maintained_acc = Accuracy()
        self.test_num_adjectives_maintained_f1 = F1()
        self.dev_tree_height_maintained_acc = Accuracy()
        self.dev_tree_height_maintained_f1 = F1()
        self.test_tree_height_maintained_acc = Accuracy()
        self.test_tree_height_maintained_f1 = F1()
        self.dev_prop_noun_maintained_acc = Accuracy()
        self.dev_prop_noun_maintained_f1 = F1()
        self.test_prop_noun_maintained_acc = Accuracy()
        self.test_prop_noun_maintained_f1 = F1()
        self.dev_dom_attr_maintained_acc = Accuracy()
        self.dev_dom_attr_maintained_f1 = F1()
        self.test_dom_attr_maintained_acc = Accuracy()
        self.test_dom_attr_maintained_f1 = F1()

        if not self.generated_sents_dir.is_dir():
            self.generated_sents_dir.mkdir(parents=True)

        self.cola_roberta_infer = ColaRobertaInfer(
            checkpoints_dir=self.cola_roberta_checkpoints_dir,
            hparams_file=self.cola_roberta_json_file,
        )

        self.label_generator = GenerateLabels(
            avg_length=10,
            src_dom_saliency_file=src_saliency_file,
            trg_dom_saliency_file=trg_saliency_file,
        )
        self.contrastive_baseline_clf_criterion = nn.BCEWithLogitsLoss()

    def forward(self):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):

        src_batch, trg_batch = batch
        (
            src_input_tensor,
            src_input_len_tensor,
            src_dec_inp_tensor,
            src_dec_output_tensor,
            _,
            src_label_tensors,
        ) = src_batch
        (
            trg_input_tensor,
            trg_input_len_tensor,
            trg_dec_inp_tensor,
            trg_dec_output_tensor,
            _,
            trg_label_tensors,
        ) = trg_batch
        src_input_len_tensor = src_input_len_tensor.squeeze(1)
        trg_input_len_tensor = trg_input_len_tensor.squeeze(1)
        src_input_len_tensor = src_input_len_tensor.cpu()
        trg_input_len_tensor = trg_input_len_tensor.cpu()

        # train trg autoencoder
        if optimizer_idx == 0 and self.autoencoder_training_off is False:

            (_, (encoder_hn, encoder_cn), decoder_logits, (_, _),) = self.trg_autoencoder(
                trg_input_tensor,
                trg_dec_inp_tensor,
                trg_input_len_tensor,
                noise=True,
                pad_idx=self.pad_idx,
            )
            plain_out = decoder_logits.view(
                decoder_logits.size(0) * decoder_logits.size(1),
                decoder_logits.size(2),
            )
            plain_tgt = trg_dec_output_tensor.view(-1)

            loss = self.lambda_ae * self.criterion_trg_ae(plain_out, plain_tgt)

            if batch_idx == self.ae_anneal_noise_every == 0:
                self.trg_autoencoder.noise_anneal(self.ae_anneal_noise)

            self.log(name="trg_enc/loss", value=loss.cpu().item())

            return {
                "loss": loss,
            }

        # train src auteoncoder
        if optimizer_idx == 1 and self.autoencoder_training_off is False:
            (_, (_, _), decoder_logits, (_, _),) = self.src_autoencoder(
                src_input_tensor,
                src_dec_inp_tensor,
                src_input_len_tensor,
                noise=True,
                pad_idx=self.pad_idx,
            )

            plain_out = decoder_logits.view(
                decoder_logits.size(0) * decoder_logits.size(1),
                decoder_logits.size(2),
            )
            plain_tgt = src_dec_output_tensor.view(-1)

            loss = self.lambda_ae * self.criterion_src_ae(plain_out, plain_tgt)

            if batch_idx == self.ae_anneal_noise_every == 0:
                self.src_autoencoder.noise_anneal(self.ae_anneal_noise)

            self.log(name="src_enc/loss", value=loss.cpu().item())

            return {
                "loss": loss,
            }

        # trg encoder adversarial training
        if optimizer_idx == 2 and self.adv_training_off is False:

            # trg-enc trg-disc adv training
            _, (real_repr, encoder_cn) = self.trg_autoencoder.encode(
                trg_input_tensor,
                trg_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )
            real_repr.register_hook(lambda grad: grad * self.grad_lambda)

            disc_out = self.trg_discriminator(real_repr)
            disc_loss = disc_out.mean()

            return {"loss": disc_loss}

        # src encoder adversarial training
        if optimizer_idx == 3 and not self.adv_training_off:

            # optimize src autoencoder as generator
            _, (fake_repr_, _) = self.src_autoencoder.encode(
                src_input_tensor,
                src_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )

            fake_disc_out = self.trg_discriminator(fake_repr_)

            if self.generator_only_contrastive_learning:
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    trg_trg_argtop,
                    trg_trg_argmin,
                    trg_src_argtop,
                    trg_src_argmin,
                    src_src_argtop,
                    src_src_argmin,
                    src_trg_argtop,
                    src_trg_argmin,
                ) = self.get_contra_batch_labelwise(src_batch=src_batch, trg_batch=trg_batch)

                # B * P * H
                fake_repr_pos = fake_repr_[src_src_argtop]

                # B * N * H
                fake_repr_neg = fake_repr_[src_src_argmin]

                contra_loss = self.contra_loss(
                    source_tensor=fake_repr_,
                    pos_batch=fake_repr_pos,
                    neg_batch=fake_repr_neg,
                    do_normalize=self.normalize_contrastive_tensors,
                )
            else:
                contra_loss = 0

            contra_loss = self.weight_contrastive_loss * contra_loss

            loss = -fake_disc_out.mean() + contra_loss

            self.log(name="src_gen/loss", value=loss.cpu().item())

            if self.generator_only_contrastive_learning:
                self.log(name="src_gen/contra_loss", value=contra_loss)

            return {"loss": loss}

        # target discriminator training
        if optimizer_idx == 4 and not self.disc_training_off:

            # *_pos_inp_tensors = batch_size, P, T
            # *_neg_inp_tensors = batch_size, N, T
            # *_pos_len_tensors = batch_size,
            # *_neg_len_tensors = batch_size,
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                trg_trg_argtop,
                trg_trg_argmin,
                trg_src_argtop,
                trg_src_argmin,
                src_src_argtop,
                src_src_argmin,
                _,
                _,
            ) = self.get_contra_batch_labelwise(src_batch=src_batch, trg_batch=trg_batch)

            # B * H
            _, (real_repr, _) = self.trg_autoencoder.encode(
                trg_input_tensor,
                trg_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )
            # B * H
            _, (fake_repr, _) = self.src_autoencoder.encode(
                src_input_tensor,
                src_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )

            # (B,1), (B, H)
            real_disc_out, real_disc_representations = self.trg_discriminator(
                real_repr.detach(), return_last=True
            )
            real_disc_representations = real_disc_representations.detach()

            # (B,1), (B, H)
            fake_disc_out, fake_disc_representations = self.trg_discriminator(
                fake_repr.detach(), return_last=True
            )
            fake_disc_representations = fake_disc_representations.detach()
            disc_loss = -(real_disc_out - fake_disc_out).mean()

            gradient_penalty = self.calc_gradient_penalty(real_repr.detach(), fake_repr.detach())

            # Use in-house contrastiveloss
            if (
                self.discriminator_contrastive_learning_off is False
                and self.use_official_sup_con_loss is False
            ):
                # trg x trg positive representations from the discriminator
                # B, P, H
                trg_domain_disc_pos = real_disc_representations[trg_trg_argtop]
                # trg x src positive representations from the discriminator
                # B, P, H
                src_domain_disc_pos = fake_disc_representations[trg_src_argtop]

                # trg x trg negatve representations from the discriminator
                # B, N,  H
                trg_domain_disc_neg = real_disc_representations[trg_trg_argmin]

                # trg x src negative representations from the discriminator
                # B, N, H
                src_domain_disc_neg = fake_disc_representations[trg_src_argmin]

                # concatenate them to calculate contrastive loss
                # B, 2P,  H
                positive_hiddens = torch.cat([trg_domain_disc_pos, src_domain_disc_pos], dim=1)
                # B, 2N, H
                negative_hiddens = torch.cat([trg_domain_disc_neg, src_domain_disc_neg], dim=1)

                contra_loss = self.contra_loss(
                    source_tensor=real_repr.detach(),
                    pos_batch=positive_hiddens.detach(),
                    neg_batch=negative_hiddens.detach(),
                    do_normalize=self.normalize_contrastive_tensors,
                )

            # Use the contrastive loss from SimCLR paper
            elif (
                self.discriminator_contrastive_learning_off is False
                and self.use_official_sup_con_loss
            ):
                # B * P * H
                trg_domain_disc_pos = real_disc_representations[trg_trg_argtop]

                # B * P * H
                src_domain_disc_pos = fake_disc_representations[trg_src_argtop]

                # combine them
                # two views of the same sentence
                # B * (P + P) * H
                features = torch.cat([trg_domain_disc_pos, src_domain_disc_pos], dim=1)

                contra_loss = self.sup_con_loss(features=features)

            else:
                contra_loss = 0.0

            if self.use_disc_attr_clf:
                inputs = torch.cat([real_repr.detach(), fake_repr.detach()], dim=0)

                # 2B * NUM_LABELS
                labels = torch.cat([trg_label_tensors, src_label_tensors], dim=0)

                # 2B * LABEL_VOCAB_SIZE
                logits = self.contrastive_baseline_classifier(inputs)

                # Make the labels a one hot vector
                labels = torch.zeros(labels.size(0), logits.size(1), device=self.device).scatter_(
                    1, labels, 1.0
                )
                clf_loss = self.contrastive_baseline_clf_criterion(logits, labels)
            else:
                clf_loss = 0

            loss = (
                disc_loss
                + gradient_penalty
                + (self.weight_contrastive_loss * contra_loss)
                + (self.weight_contrastive_loss * clf_loss)
            )

            #########################
            # LOGGING
            #########################
            self.log(name="contrastive/disc_loss", value=contra_loss)
            self.log(name="disc_trg/loss", value=loss.cpu().item())

            return {
                "loss": loss,
            }

        # Encoder contrastive learning
        if optimizer_idx == 5:
            # B * H
            _, (real_repr, _) = self.trg_autoencoder.encode(
                inp=trg_input_tensor,
                lengths=trg_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )

            # B * H
            _, (fake_repr, _) = self.src_autoencoder.encode(
                inp=src_input_tensor,
                lengths=src_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )

            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                trg_trg_argtop,
                trg_trg_argmin,
                trg_src_argtop,
                trg_src_argmin,
                src_src_argtop,
                src_src_argmin,
                src_trg_argtop,
                src_trg_argmin,
            ) = self.get_contra_batch_labelwise(src_batch=src_batch, trg_batch=trg_batch)

            src_domain_enc_pos = fake_repr[src_src_argtop]
            trg_domain_enc_pos = real_repr[src_trg_argtop]
            src_domain_enc_neg = fake_repr[src_src_argmin]
            trg_domain_enc_neg = real_repr[src_trg_argmin]

            if (
                self.encoder_contrastive_learning_off is False
                and self.use_official_sup_con_loss is True
            ):
                features = torch.cat([trg_domain_enc_pos, src_domain_enc_pos], dim=1)
                contra_loss = self.sup_con_loss(features=features)
            elif (
                self.encoder_contrastive_learning_off is False
                and self.use_official_sup_con_loss is False
            ):
                # concatenate the representations to calculate the loss
                pos_hiddens = torch.cat([trg_domain_enc_pos, src_domain_enc_pos], dim=1)
                neg_hiddens = torch.cat([trg_domain_enc_neg, src_domain_enc_neg], dim=1)

                contra_loss = self.contra_loss(
                    source_tensor=real_repr,
                    pos_batch=pos_hiddens,
                    neg_batch=neg_hiddens,
                    do_normalize=self.normalize_contrastive_tensors,
                )
            else:
                contra_loss = 0

            if self.use_enc_attr_clf:
                # 2B * H
                discriminator_inputs = torch.cat([real_repr, fake_repr], dim=0)

                # 2B * num_classes
                logits = self.contrastive_baseline_classifier(discriminator_inputs)

                # 2B * num_classes
                labels = torch.cat([trg_label_tensors, src_label_tensors], dim=0)

                # Make the labels a one hot vector
                labels = torch.zeros(
                    src_label_tensors.size(0) + trg_label_tensors.size(0),
                    logits.size(1),
                    device=self.device,
                ).scatter_(1, labels, 1.0)

                clf_loss = self.contrastive_baseline_clf_criterion(logits, labels)

            else:
                clf_loss = 0

            loss = self.weight_contrastive_loss * contra_loss + (
                self.weight_contrastive_loss * clf_loss
            )

            self.log(name="contrastive/encoder_loss", value=contra_loss)
            self.log(name="contrastive/baseline_attr_clf_loss", value=clf_loss)

            return {"loss": loss}

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ):

        # trg autoencoder training
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()

        # src autoencoder training
        if optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()

        # trg encoder adversarial training
        if optimizer_idx == 2:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()

        # src encoder adversarial training
        if optimizer_idx == 3:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()

        # trg_discriminator training
        if optimizer_idx == 4:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()

        if optimizer_idx == 5:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()

    def configure_optimizers(self):
        optim_src_ae = optim.Adam(
            self.src_autoencoder.parameters(),
            lr=self.lr_gen,
            betas=(0.5, 0.999),
        )

        if self.single_encoder_two_decoders:
            optim_trg_ae = optim.Adam(self.trg_autoencoder.parameters(), lr=self.lr_ae)
        else:
            optim_trg_ae = optim.SGD(self.trg_autoencoder.parameters(), lr=self.lr_ae)

        disc_parameters = [self.trg_discriminator.parameters()]

        if self.use_disc_attr_clf:
            disc_parameters.append(self.contrastive_baseline_classifier.parameters())

        optim_trg_disc = optim.Adam(
            itertools.chain(*disc_parameters),
            lr=self.lr_disc,
            betas=(0.5, 0.999),
        )
        optimizers = [
            {"optimizer": optim_trg_ae, "frequency": 1},  # for autoenc train
            {"optimizer": optim_src_ae, "frequency": 1},  # for autoenc train
            {"optimizer": optim_trg_ae, "frequency": 1},  # for adv train
            {"optimizer": optim_src_ae, "frequency": 1},  # for adv train
            {
                "optimizer": optim_trg_disc,
                "frequency": self.disc_num_steps,
            },  # for disc train
        ]

        if self.encoder_contrastive_learning_off is False or self.use_enc_attr_clf is True:
            parameters = [
                self.trg_autoencoder.parameters(),
                self.src_autoencoder.parameters(),
            ]

            # if you are using the baseline classifier that classifies the attributes
            # optimizer those parameters as well
            if self.use_enc_attr_clf:
                parameters.append(self.contrastive_baseline_classifier.parameters())

            contrastive_optim = optim.Adam(
                itertools.chain(*parameters),
                lr=self.lr_gen,
                betas=(0.5, 0.999),
            )
            optimizers.append({"optimizer": contrastive_optim, "frequency": 1})

        return optimizers

    def train_dataloader(self):
        return self.dm.train_dataloader()

    def val_dataloader(self):
        return self.dm.val_dataloader()

    def test_dataloader(self):
        return self.dm.test_dataloader()

    @staticmethod
    def _build_cross_entropy_loss(pad_idx):
        return torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    def calc_gradient_penalty(self, real_data, fake_data):
        batch_size, h_size = real_data.size()
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, h_size).to(self.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        disc_interpolates = self.trg_discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gan_gp_lambda
        return gradient_penalty

    def validation_step(self, batch, batch_idx, dataloader_idx):

        # source data loader
        (
            input_tensor,
            input_len_tensor,
            dec_inp_tensor,
            dec_output_tensor,
            mask_tensor,
            label_tensor,
        ) = batch
        input_len_tensor = input_len_tensor.cpu()
        input_len_tensor = input_len_tensor.squeeze(1)  # B

        if dataloader_idx == 0:
            dev_sentences = tensors_to_text(tensors=input_tensor, idx2token=self.idx2word)

            if self.generate_through_trg:
                (
                    encoder_outputs,
                    (fake_repr, encoder_cn),
                    decoder_logits,
                    _,
                ) = self.trg_autoencoder(
                    input_tensor,
                    dec_inp_tensor,
                    input_len_tensor,
                    noise=False,
                    pad_idx=self.pad_idx,
                )
            else:
                (
                    encoder_outputs,
                    (fake_repr, encoder_cn),
                    decoder_logits,
                    _,
                ) = self.src_autoencoder(
                    input_tensor,
                    dec_inp_tensor,
                    input_len_tensor,
                    noise=False,
                    pad_idx=self.pad_idx,
                )

            ############################
            # CALCULATE LOSS
            ############################
            plain_out = decoder_logits.view(
                decoder_logits.size(0) * decoder_logits.size(1),
                decoder_logits.size(2),
            )
            plain_tgt = dec_output_tensor.view(-1)
            loss = self.criterion_src_ae(plain_out, plain_tgt)

            #############################
            # DECODE TO TARGET SIDE
            #############################
            generated_idxs = self.trg_autoencoder.generate(
                encoder_outputs,
                (fake_repr, encoder_cn),
                fake_repr,
                self.sos_idx,
                self.eos_idx,
                self.max_seq_length,
            )
            generated_sentences = tensors_to_text(generated_idxs, self.idx2word)

            table_data = list(zip(dev_sentences, generated_sentences))
            self.logger.experiment.log(
                {"Dev Translations": wandb.Table(data=table_data, columns=["Src", "Trg"])},
                commit=False,
            )

            # target sentences should be labelled with __label__2 in fast text
            # for the model
            # returns predictions, probabilities. Taking only predictions
            domain_predictions = self.ft_model.predict(generated_sentences)[0]
            # For every line the prediction can contain multiple predictions
            # Since ours is single prediction per line, take the first element
            domain_predictions = list(map(lambda pred: pred[0], domain_predictions))
            domain_predictions = [self.ft_style_mapping[pred] for pred in domain_predictions]
            domain_predictions = torch.LongTensor(domain_predictions)
            true_labels = torch.LongTensor([1] * len(domain_predictions))
            acc = self.dev_transfer_accuracy(domain_predictions, true_labels)

            return {
                "loss": loss.cpu().item(),
                "generated_sents": generated_sentences,
                "dev_sents": dev_sentences,
                "transfer_acc": acc.item(),
                "domain_predictions": domain_predictions.tolist(),
            }

        # target data loader
        if dataloader_idx == 1:
            _, (_, _), decoder_logits, (_, _) = self.trg_autoencoder(
                input_tensor,
                dec_inp_tensor,
                input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )
            plain_out = decoder_logits.view(
                decoder_logits.size(0) * decoder_logits.size(1),
                decoder_logits.size(2),
            )
            plain_tgt = dec_output_tensor.view(-1)
            loss = self.criterion_trg_ae(plain_out, plain_tgt)

            return {"loss": loss.cpu().item()}

    def validation_epoch_end(self, outs):

        src_losses = []
        trg_losses = []
        generated_sents = []
        dev_sents = []
        transfer_accs = []
        domain_predictions = []

        for idx, dataloader_output_result in enumerate(outs):
            if idx == 0:
                for out in dataloader_output_result:
                    loss = out["loss"]
                    generated_sents_ = out["generated_sents"]
                    dev_sents_ = out["dev_sents"]
                    domain_predictions_ = out["domain_predictions"]
                    trg_losses.append(loss)
                    transfer_accs.append(out["transfer_acc"])
                    generated_sents.extend(generated_sents_)
                    dev_sents.extend(dev_sents_)
                    domain_predictions.extend(domain_predictions_)

            if idx == 1:
                for out in dataloader_output_result:
                    loss = out["loss"]
                    src_losses.append(loss)

        sim_scores = self.semantic_sim.find_similarity(dev_sents, generated_sents)

        # infer using cola roberta based model
        acceptable_preds = self.cola_roberta_infer.predict(generated_sents).tolist()
        percent_acceptable = np.mean(acceptable_preds)

        avg_src_dev_loss = np.mean(src_losses)
        avg_trg_dev_loss = np.mean(trg_losses)
        avg_transfer_accs = np.mean(transfer_accs)

        avg_sim_score = np.mean(sim_scores)

        dev_sentence_lengths = []
        dev_personal_pronouns = []
        dev_descriptives = []
        dev_tree_height_maintained = []
        dev_prop_noun_maintained = []
        dev_num_dom_spec_maintained = []

        for line in self.label_generator.from_lines(lines=dev_sents):
            length_ = line["length"]
            personal_ = line["personal"]
            descriptive_ = line["descriptive"]
            tree_height = line["tree_height"]
            prop_noun = line["prop_noun"]
            dom_attrs = line["num_dom_attributes"]
            length_idx = self.labelword2idx[length_]
            personal_idx = self.labelword2idx[personal_]
            descriptive_idx = self.labelword2idx[descriptive_]
            tree_height_idx = self.labelword2idx[tree_height]
            prop_noun_idx = self.labelword2idx[prop_noun]
            dom_attrs_idx = self.labelword2idx[dom_attrs]
            dev_sentence_lengths.append(length_idx)
            dev_personal_pronouns.append(personal_idx)
            dev_descriptives.append(descriptive_idx)
            dev_tree_height_maintained.append(tree_height_idx)
            dev_prop_noun_maintained.append(prop_noun_idx)
            dev_num_dom_spec_maintained.append(dom_attrs_idx)

        generated_sentence_lengths = []
        generated_personal_pronouns = []
        generated_descriptives = []
        generated_tree_heights = []
        generated_prop_nouns = []
        generated_dom_spec_maintained = []

        for line in self.label_generator.from_lines(lines=generated_sents):
            length_ = line["length"]
            personal_ = line["personal"]
            descriptive_ = line["descriptive"]
            tree_height = line["tree_height"]
            prop_noun = line["prop_noun"]
            dom_attrs = line["num_dom_attributes"]
            length_idx = self.labelword2idx[length_]
            personal_idx = self.labelword2idx[personal_]
            descriptive_idx = self.labelword2idx[descriptive_]
            tree_height_idx = self.labelword2idx[tree_height]
            prop_noun_idx = self.labelword2idx[prop_noun]
            dom_attrs_idx = self.labelword2idx[dom_attrs]
            generated_sentence_lengths.append(length_idx)
            generated_personal_pronouns.append(personal_idx)
            generated_descriptives.append(descriptive_idx)
            generated_tree_heights.append(tree_height_idx)
            generated_prop_nouns.append(prop_noun_idx)
            generated_dom_spec_maintained.append(dom_attrs_idx)

        len_maintained_acc = self.dev_len_maintained_acc(
            target=torch.LongTensor(dev_sentence_lengths),
            preds=torch.LongTensor(generated_sentence_lengths),
        )

        len_maintained_f1 = self.dev_len_maintained_f1(
            target=torch.LongTensor(dev_sentence_lengths),
            preds=torch.LongTensor(generated_sentence_lengths),
        )

        personal_pron_maintaned_acc = self.dev_pron_maintained_acc(
            target=torch.LongTensor(dev_personal_pronouns),
            preds=torch.LongTensor(generated_personal_pronouns),
        )

        personal_pron_maintained_f1 = self.dev_pron_maintained_f1(
            target=torch.LongTensor(dev_personal_pronouns),
            preds=torch.LongTensor(generated_personal_pronouns),
        )

        descriptive_maintained_acc = self.dev_num_adjectives_maintained_acc(
            target=torch.LongTensor(dev_descriptives),
            preds=torch.LongTensor(generated_descriptives),
        )

        descriptive_maintained_f1 = self.dev_num_adjectives_maintained_f1(
            target=torch.LongTensor(dev_descriptives),
            preds=torch.LongTensor(generated_descriptives),
        )

        tree_height_maintained_acc = self.dev_tree_height_maintained_acc(
            target=torch.LongTensor(dev_tree_height_maintained),
            preds=torch.LongTensor(generated_tree_heights),
        )

        tree_height_maintained_f1 = self.dev_tree_height_maintained_f1(
            target=torch.LongTensor(dev_tree_height_maintained),
            preds=torch.LongTensor(generated_tree_heights),
        )

        prop_noun_maintained_acc = self.dev_prop_noun_maintained_acc(
            target=torch.LongTensor(dev_prop_noun_maintained),
            preds=torch.LongTensor(generated_prop_nouns),
        )

        prop_noun_maintained_f1 = self.dev_prop_noun_maintained_f1(
            target=torch.LongTensor(dev_prop_noun_maintained),
            preds=torch.LongTensor(generated_prop_nouns),
        )

        dev_domspecattr_maintained_acc = self.dev_dom_attr_maintained_acc(
            target=torch.LongTensor(dev_num_dom_spec_maintained),
            preds=torch.LongTensor(generated_dom_spec_maintained),
        )

        dev_domspecattr_maintained_f1 = self.dev_dom_attr_maintained_f1(
            target=torch.LongTensor(dev_num_dom_spec_maintained),
            preds=torch.LongTensor(generated_dom_spec_maintained),
        )

        # instance wise metric calculation
        assert len(domain_predictions) == len(acceptable_preds) == len(sim_scores)

        ############################################
        # Calculate instance wise aggregate metrics
        ############################################
        aggregate = torch.mul(torch.Tensor(domain_predictions), torch.Tensor(acceptable_preds))
        aggregate = torch.mul(aggregate, torch.Tensor(sim_scores))
        dev_aggregate = torch.mean(aggregate)

        self.log(name="dev_agg", value=dev_aggregate)
        self.log(name="src_enc/val_loss", value=avg_src_dev_loss)
        self.log(name="trg_enc/val_loss", value=avg_trg_dev_loss)
        self.log(name="dev_acceptability", value=percent_acceptable)
        self.log(name="dev_sim", value=avg_sim_score)
        self.log(name="dev_transfer_acc", value=avg_transfer_accs)
        self.log(name="dev_len_maintain_acc", value=len_maintained_acc)
        self.log(name="dev_len_maintain_f1", value=len_maintained_f1)
        self.log(name="dev_personal_pron_maintained_acc", value=personal_pron_maintaned_acc)
        self.log(name="dev_personal_pron_maintained_f1", value=personal_pron_maintained_f1)
        self.log(name="dev_descriptive_maintained_acc", value=descriptive_maintained_acc)
        self.log(name="dev_descriptive_maintained_f1", value=descriptive_maintained_f1)
        self.log(name="dev_tree_height_maintained_acc", value=tree_height_maintained_acc)
        self.log(name="dev_tree_height_maintained_f1", value=tree_height_maintained_f1)
        self.log(name="dev_prop_noun_maintained_acc", value=prop_noun_maintained_acc)
        self.log(name="dev_prop_noun_maintained_f1", value=prop_noun_maintained_f1)
        self.log(name="dev_domspecattr_maintained_acc", value=dev_domspecattr_maintained_acc)
        self.log(name="dev_domspecattr_maintained_f1", value=dev_domspecattr_maintained_f1)

        generated_sents_file = self.generated_sents_dir.joinpath(f"{self.current_epoch}.txt")
        with open(str(generated_sents_file), "w") as fp:
            for dev_sent, gen_sent in zip(dev_sents, generated_sents):
                fp.write(f"{dev_sent}\n{gen_sent}\n")
                fp.write("=" * 80)
                fp.write("\n")

    def test_step(self, batch, batch_idx, dataloader_idx):
        test_sentences = []
        generated_sentences = []
        if dataloader_idx == 0:
            (
                input_tensor,
                input_len_tensor,
                dec_inp_tensor,
                dec_output_tensor,
                mask_tensor,
                label_input_tensors,
            ) = batch
            test_sentences_ = tensors_to_text(tensors=input_tensor.cpu(), idx2token=self.idx2word)
            test_sentences.extend(test_sentences_)
            input_len_tensor = input_len_tensor.cpu().squeeze(1)  # B

            if self.generate_through_trg:
                (encoder_outputs, (fake_repr, encoder_cn),) = self.trg_autoencoder.encode(
                    input_tensor,
                    input_len_tensor,
                    noise=False,
                    pad_idx=self.pad_idx,
                )
            else:
                (encoder_outputs, (fake_repr, encoder_cn),) = self.src_autoencoder.encode(
                    input_tensor,
                    input_len_tensor,
                    noise=False,
                    pad_idx=self.pad_idx,
                )

            generated_idxs = self.trg_autoencoder.generate(
                encoder_outputs,
                (fake_repr, encoder_cn),
                fake_repr,
                self.sos_idx,
                self.eos_idx,
                self.max_seq_length,
            )

            generated_sentences_ = tensors_to_text(generated_idxs, self.idx2word)
            generated_sentences.extend(generated_sentences_)

        return {
            "test_sentences": test_sentences,
            "generated_sentences": generated_sentences,
        }

    def test_epoch_end(self, outs: List[Any]):

        test_sents = []
        generated_sents = []

        for idx, dataloader_output_result in enumerate(outs):
            if idx == 0:
                for out in dataloader_output_result:
                    test_sentences = out["test_sentences"]
                    generated_sentences = out["generated_sentences"]
                    test_sents.extend(test_sentences)
                    generated_sents.extend(generated_sentences)

        sim_scores = self.semantic_sim.find_similarity(test_sents, generated_sents)
        avg_sim_score = np.mean(sim_scores)

        # infer using cola roberta based model
        acceptable_preds = self.cola_roberta_infer.predict(generated_sents).tolist()
        percent_acceptable = np.mean(acceptable_preds)

        # target sentences should be labelled with __label__2 in fast text
        # for the model
        # returns predictions, probabilities. Taking only predictions
        domain_predictions, _ = self.ft_model.predict(generated_sents)
        # For every line the prediction can contain multiple predictions
        # Since ours is single prediction per line, take the first element
        domain_predictions = list(map(lambda pred: pred[0], domain_predictions))
        domain_predictions = [self.ft_style_mapping[pred] for pred in domain_predictions]
        domain_predictions = torch.LongTensor(domain_predictions)
        true_labels = torch.LongTensor([1] * len(domain_predictions))
        transfer_acc = self.test_transfer_accuracy(domain_predictions, true_labels)

        aggregate = torch.mul(domain_predictions.float(), torch.Tensor(acceptable_preds))
        aggregate = torch.mul(aggregate, torch.Tensor(sim_scores))
        test_aggregate = torch.mean(aggregate)

        test_sentence_lengths = []
        test_personal_prons = []
        test_descriptives = []
        test_tree_height_maintained = []
        test_prop_noun_maintained = []
        test_num_dom_spec_maintained = []

        for line in self.label_generator.from_lines(lines=test_sents):
            length_ = line["length"]
            personal_ = line["personal"]
            descriptive_ = line["descriptive"]
            tree_height = line["tree_height"]
            prop_noun = line["prop_noun"]
            dom_attrs = line["num_dom_attributes"]

            length_idx = self.labelword2idx[length_]
            personal_idx = self.labelword2idx[personal_]
            descriptive_idx = self.labelword2idx[descriptive_]
            tree_height_idx = self.labelword2idx[tree_height]
            prop_noun_idx = self.labelword2idx[prop_noun]
            dom_attrs_idx = self.labelword2idx[dom_attrs]

            test_sentence_lengths.append(length_idx)
            test_personal_prons.append(personal_idx)
            test_descriptives.append(descriptive_idx)
            test_tree_height_maintained.append(tree_height_idx)
            test_prop_noun_maintained.append(prop_noun_idx)
            test_num_dom_spec_maintained.append(dom_attrs_idx)

        generated_sentence_lengths = []
        generated_personal_prons = []
        generated_descriptives = []
        generated_tree_heights = []
        generated_prop_nouns = []
        generated_dom_spec_maintained = []

        for line in self.label_generator.from_lines(lines=generated_sents):
            length_ = line["length"]
            personal_ = line["personal"]
            descriptive_ = line["descriptive"]
            tree_height = line["tree_height"]
            prop_noun = line["prop_noun"]
            dom_attrs = line["num_dom_attributes"]

            length_idx = self.labelword2idx[length_]
            personal_idx = self.labelword2idx[personal_]
            descriptive_idx = self.labelword2idx[descriptive_]
            tree_height_idx = self.labelword2idx[tree_height]
            prop_noun_idx = self.labelword2idx[prop_noun]
            dom_attrs_idx = self.labelword2idx[dom_attrs]

            generated_sentence_lengths.append(length_idx)
            generated_personal_prons.append(personal_idx)
            generated_descriptives.append(descriptive_idx)
            generated_tree_heights.append(tree_height_idx)
            generated_prop_nouns.append(prop_noun_idx)
            generated_dom_spec_maintained.append(dom_attrs_idx)

        len_maintained_acc = self.test_len_maintained_acc(
            target=torch.LongTensor(test_sentence_lengths),
            preds=torch.LongTensor(generated_sentence_lengths),
        )

        len_maintained_f1 = self.test_len_maintained_f1(
            target=torch.LongTensor(test_sentence_lengths),
            preds=torch.LongTensor(generated_sentence_lengths),
        )

        test_personal_pron_maintaned_acc = self.test_pron_maintained_acc(
            target=torch.LongTensor(test_personal_prons),
            preds=torch.LongTensor(generated_personal_prons),
        )

        test_personal_pron_maintaned_f1 = self.test_pron_maintained_f1(
            target=torch.LongTensor(test_personal_prons),
            preds=torch.LongTensor(generated_personal_prons),
        )

        descriptive_maintained_acc = self.dev_num_adjectives_maintained_acc(
            target=torch.LongTensor(test_descriptives),
            preds=torch.LongTensor(generated_descriptives),
        )

        descriptive_maintained_f1 = self.dev_num_adjectives_maintained_f1(
            target=torch.LongTensor(test_descriptives),
            preds=torch.LongTensor(generated_descriptives),
        )

        tree_height_maintained_acc = self.dev_tree_height_maintained_acc(
            target=torch.LongTensor(test_tree_height_maintained),
            preds=torch.LongTensor(generated_tree_heights),
        )

        tree_height_maintained_f1 = self.dev_tree_height_maintained_f1(
            target=torch.LongTensor(test_tree_height_maintained),
            preds=torch.LongTensor(generated_tree_heights),
        )

        prop_noun_maintained_acc = self.dev_prop_noun_maintained_acc(
            target=torch.LongTensor(test_prop_noun_maintained),
            preds=torch.LongTensor(generated_prop_nouns),
        )

        prop_noun_maintained_f1 = self.dev_prop_noun_maintained_f1(
            target=torch.LongTensor(test_prop_noun_maintained),
            preds=torch.LongTensor(generated_prop_nouns),
        )

        domspecattr_maintained_acc = self.dev_dom_attr_maintained_acc(
            target=torch.LongTensor(test_num_dom_spec_maintained),
            preds=torch.LongTensor(generated_dom_spec_maintained),
        )

        domspecattr_maintained_f1 = self.dev_dom_attr_maintained_f1(
            target=torch.LongTensor(test_num_dom_spec_maintained),
            preds=torch.LongTensor(generated_dom_spec_maintained),
        )
        self.log(name="test_agg", value=test_aggregate)
        self.log(name="test_sim", value=avg_sim_score)
        self.log(name="test_percent_acceptable", value=percent_acceptable)
        self.log(name="test_len_maintained_acc", value=len_maintained_acc)
        self.log(name="test_len_maintained_f1", value=len_maintained_f1)
        self.log(name="test_pron_maintained_acc", value=test_personal_pron_maintaned_acc)
        self.log(name="test_pron_maintained_f1", value=test_personal_pron_maintaned_f1)
        self.log(name="test_descriptive_maintained_acc", value=descriptive_maintained_acc)
        self.log(name="test_descriptive_maintained_f1", value=descriptive_maintained_f1)
        self.log(name="test_tree_height_maintained_acc", value=tree_height_maintained_acc)
        self.log(name="test_tree_height_maintained_f1", value=tree_height_maintained_f1)
        self.log(name="test_prop_noun_maintained_acc", value=prop_noun_maintained_acc)
        self.log(name="test_prop_noun_maintained_f1", value=prop_noun_maintained_f1)
        self.log(name="test_domspecattr_maintained_acc", value=domspecattr_maintained_acc)
        self.log(name="test_domspecattr_maintained_f1", value=domspecattr_maintained_f1)

        self.logger.experiment.log(
            {
                "Test Translations": wandb.Table(
                    data=list(zip(test_sents, generated_sents)),
                    columns=["Src", "Trg"],
                )
            },
        )

    def get_contra_batch_labelwise(self, src_batch, trg_batch):
        """Positive examples are those that belong to the
        same label as the data point. This is irrespective of whether it comes
        from the source or the target domain. The similarity
        between the points with respect to the labels are calculated
        between the target and the source samples.

        Parameters
        ----------
        src_batch: List[torch.Tensor]
        trg_batch: List[torch.Tensor]

        """
        (
            src_input_tensor,
            src_input_len_tensor,
            src_dec_inp_tensor,
            src_dec_output_tensor,
            _,
            src_label_tensors,
        ) = src_batch
        (
            trg_input_tensor,
            trg_input_len_tensor,
            trg_dec_inp_tensor,
            trg_dec_output_tensor,
            _,
            trg_label_tensors,
        ) = trg_batch

        src_input_len_tensor = src_input_len_tensor.squeeze(1).cpu()
        trg_input_len_tensor = trg_input_len_tensor.squeeze(1).cpu()

        src_label_tensors = src_label_tensors.float()
        trg_label_tensors = trg_label_tensors.float()

        src_dom_bsz = src_label_tensors.size(0)
        trg_dom_bsz = trg_label_tensors.size(0)

        # similarity of trg domain samples with trg domain samples
        trg_trg_sim = pairwise_attr_intersection(
            first_batch=trg_label_tensors,
            second_batch=trg_label_tensors,
            device=self.device,
        )

        if self.num_nearest_labels > trg_trg_sim.size(1):
            k = trg_trg_sim.size(1) - 1

        else:
            k = self.num_nearest_labels

        _, trg_trg_argtop = torch.topk(trg_trg_sim, k=k, largest=True, dim=1)
        _, trg_trg_argmin = torch.topk(
            trg_trg_sim, k=trg_dom_bsz - self.num_nearest_labels, largest=False, dim=1
        )

        # trg domain postives and negatives from the same domain
        # *_pos_inp_tensors = B, P, T
        # *_neg_inp_tensors = B, N , T
        # *_pos_len_tensors = B, P
        # *_neg_len_tensors = B, N
        trg_trg_pos_inp_tensors = trg_input_tensor[trg_trg_argtop]
        trg_trg_neg_inp_tensors = trg_input_tensor[trg_trg_argmin]
        trg_trg_pos_len_tensors = trg_input_len_tensor[trg_trg_argtop]
        trg_trg_neg_len_tensors = trg_input_len_tensor[trg_trg_argmin]

        # similarity of trg domain samples with src domain samples
        trg_src_sim = pairwise_attr_intersection(
            first_batch=trg_label_tensors,
            second_batch=src_label_tensors,
            device=self.device,
        )

        _, trg_src_argtop = torch.topk(trg_src_sim, k=self.num_nearest_labels, largest=True, dim=1)
        _, trg_src_argmin = torch.topk(
            trg_src_sim, k=src_dom_bsz - self.num_nearest_labels, largest=False, dim=1
        )

        trg_src_pos_inp_tensors = src_input_tensor[trg_src_argtop]
        trg_src_neg_inp_tensors = src_input_tensor[trg_src_argmin]
        trg_src_pos_len_tensors = src_input_len_tensor[trg_src_argtop]
        trg_src_neg_len_tensors = src_input_len_tensor[trg_src_argmin]

        src_src_sim = pairwise_attr_intersection(
            first_batch=src_label_tensors,
            second_batch=src_label_tensors,
            device=self.device,
        )
        _, src_src_argtop = torch.topk(src_src_sim, k=self.num_nearest_labels, largest=True, dim=1)
        _, src_src_argmin = torch.topk(
            src_src_sim, k=src_dom_bsz - self.num_nearest_labels, largest=False, dim=1
        )

        src_src_pos_inp_tensors = src_input_tensor[src_src_argtop]
        src_src_neg_inp_tensors = src_input_tensor[src_src_argmin]
        src_src_pos_len_tensors = src_input_len_tensor[src_src_argtop]
        src_src_neg_len_tensors = src_input_len_tensor[src_src_argmin]

        src_trg_sim = pairwise_attr_intersection(
            first_batch=src_label_tensors,
            second_batch=trg_label_tensors,
            device=self.device,
        )

        _, src_trg_argtop = torch.topk(src_trg_sim, k=self.num_nearest_labels, largest=True, dim=1)

        _, src_trg_argmin = torch.topk(
            src_trg_sim, k=src_dom_bsz - self.num_nearest_labels, largest=False, dim=1
        )

        src_trg_pos_inp_tensors = trg_input_tensor[src_trg_argtop]
        src_trg_neg_inp_tensors = trg_input_tensor[src_trg_argmin]
        src_trg_pos_len_tensors = trg_input_len_tensor[src_trg_argtop]
        src_trg_neg_len_tensors = trg_input_len_tensor[src_trg_argmin]

        return (
            trg_trg_pos_inp_tensors,
            trg_trg_neg_inp_tensors,
            trg_trg_pos_len_tensors,
            trg_trg_neg_len_tensors,
            trg_src_pos_inp_tensors,
            trg_src_neg_inp_tensors,
            trg_src_pos_len_tensors,
            trg_src_neg_len_tensors,
            src_src_pos_inp_tensors,
            src_src_neg_inp_tensors,
            src_src_pos_len_tensors,
            src_src_neg_len_tensors,
            src_trg_pos_inp_tensors,
            src_trg_neg_inp_tensors,
            src_trg_pos_len_tensors,
            src_trg_neg_len_tensors,
            trg_trg_argtop,
            trg_trg_argmin,
            trg_src_argtop,
            trg_src_argmin,
            src_src_argtop,
            src_src_argmin,
            src_trg_argtop,
            src_trg_argmin,
        )
