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
from dct.models.cycle_mapper import CycleMapper
import torch.nn as nn
from torch.optim import Optimizer
import fasttext
from pytorch_lightning.metrics import Accuracy
from dct.evaluation.sim.SIM import SIM
from dct.infer.cola_roberta_infer import ColaRobertaInfer


class PLARAE(pl.LightningModule):
    def __init__(
        self,
        src_autoencoder: Seq2Seq,
        trg_autoencoder: Seq2Seq,
        trg_discriminator: Critic,
        mapper_F: Optional[CycleMapper],
        mapper_G: Optional[CycleMapper],
        src_discriminator: Optional[Critic],
        datamodule: DataModule,
        ft_model,
        hparams: Dict[str, Any],
        cola_roberta_checkpoints_dir: Optional[pathlib.Path] = None,
        cola_roberta_json_file: Optional[pathlib.Path] = None,
    ):
        """

        :type ft_model: fasttext._FastText
        """
        super(PLARAE, self).__init__()
        self.save_hyperparameters(hparams)
        self.src_autoencoder = src_autoencoder
        self.trg_autoencoder = trg_autoencoder
        self.trg_discriminator = trg_discriminator
        self.dm = datamodule
        self.ft_model = ft_model
        self.dm.prepare_data()
        self.word2idx = self.dm.vocab.get_word2idx()
        self.idx2word = self.dm.vocab.get_idx2word()
        self.pad_idx = self.word2idx["<pad>"]
        self.sos_idx = self.word2idx["<s>"]
        self.eos_idx = self.word2idx["</s>"]
        self.criterion_src_ae = self._build_cross_entropy_loss(pad_idx=self.pad_idx)
        self.criterion_trg_ae = self._build_cross_entropy_loss(pad_idx=self.pad_idx)
        self.cola_roberta_checkpoints_dir = cola_roberta_checkpoints_dir
        self.cola_roberta_json_file = cola_roberta_json_file
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
        self.max_seq_length = hparams["max_seq_length"]
        self.gan_gp_lambda = hparams["gan_gp_lambda"]
        self.grad_lambda = hparams["grad_lambda"]
        self.autoencoder_training_off = hparams["autoencoder_training_off"]
        self.disc_training_off = hparams["disc_training_off"]
        self.adv_training_off = hparams["adv_training_off"]
        self.dev_off = hparams["devoff"]
        self.disc_num_steps = hparams["disc_num_steps"]
        self.train_gen_num_samples = 1000  # HARDCODED FOR NOW
        self.wasabi_print = wasabi.Printer()
        self.generated_sents_dir = self.exp_dir.joinpath("generated_sents")
        self.generate_through_trg = hparams["generate_through_trg"]
        self.enable_cycle_const = hparams["enable_cycle_consistency"]
        self.single_encoder_two_decoders = hparams["single_encoder_two_decoders"]
        self.mapper_F = mapper_F
        self.mapper_G = mapper_G
        self.src_discriminator = src_discriminator
        self.mapper_lr = hparams["mapper_lr"]
        self.ft_style_mapping = {"__label__1": 0, "__label__2": 1}
        self.lambda_ae = hparams["lambda_ae"]
        self.lambda_cc = hparams["lambda_cc"]
        self.lambda_adv = hparams["lambda_adv"]
        self.dev_transfer_accuracy = Accuracy()
        self.test_transfer_accuracy = Accuracy()

        if not self.generated_sents_dir.is_dir():
            self.generated_sents_dir.mkdir(parents=True)

        if self.enable_cycle_const:
            assert mapper_F is not None
            assert mapper_G is not None
            self.cyc_loss = nn.L1Loss()
            self.optimizer_mapper_F = optim.Adam(
                mapper_F.parameters(), lr=self.mapper_lr
            )
            self.optimizer_mapper_G = optim.Adam(
                mapper_G.parameters(), lr=self.mapper_lr
            )

        self.cola_roberta_infer = ColaRobertaInfer(
            checkpoints_dir=self.cola_roberta_checkpoints_dir,
            hparams_file=self.cola_roberta_json_file,
        )

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
            _,
        ) = src_batch
        (
            trg_input_tensor,
            trg_input_len_tensor,
            trg_dec_inp_tensor,
            trg_dec_output_tensor,
            _,
            _,
        ) = trg_batch
        src_input_len_tensor = src_input_len_tensor.squeeze(1)
        trg_input_len_tensor = trg_input_len_tensor.squeeze(1)
        src_input_len_tensor = src_input_len_tensor.cpu()
        trg_input_len_tensor = trg_input_len_tensor.cpu()

        # train trg autoencoder
        if optimizer_idx == 0 and not self.autoencoder_training_off:

            (
                _,
                (encoder_hn, encoder_cn),
                decoder_logits,
                (_, _),
            ) = self.trg_autoencoder(
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

            self.log(
                "trg_enc/loss",
                value=loss.cpu().item(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )

            return {
                "loss": loss,
            }

        # train src auteoncoder
        if optimizer_idx == 1 and not self.autoencoder_training_off:
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

            self.log(
                name="src_enc/loss",
                value=loss.cpu().item(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )

            return {
                "loss": loss,
            }

        # trg encoder adversarial training
        if optimizer_idx == 2 and not self.adv_training_off:

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

            if self.enable_cycle_const:
                cycle_loss = self.cyc_loss(
                    self.mapper_F(self.mapper_G(real_repr)), real_repr
                )
                self.log(
                    name="trg_enc/cyc_loss",
                    value=cycle_loss.item(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

                cycle_loss = self.lambda_cc * cycle_loss
                disc_loss += cycle_loss

            # trg-enc as generator of fake samples for src-disc
            if self.src_discriminator is not None:
                assert self.mapper_G is not None
                assert self.enable_cycle_const
                fake_repr = real_repr
                fake_repr = self.mapper_G(fake_repr)

                src_disc_out = self.src_discriminator(fake_repr)
                src_disc_loss = -src_disc_out.mean()
                disc_loss += src_disc_loss
                self.log(
                    name=f"trg_gen/loss",
                    value=src_disc_loss.cpu().item(),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )

            return {"loss": disc_loss}

        if optimizer_idx == 3 and not self.adv_training_off:

            # optimize src autoencoder as generator
            _, (fake_repr_, _) = self.src_autoencoder.encode(
                src_input_tensor,
                src_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )

            if self.enable_cycle_const:
                fake_repr = self.mapper_F(fake_repr_)
                fake_disc_out = self.trg_discriminator(fake_repr)
            else:
                fake_disc_out = self.trg_discriminator(fake_repr_)

            loss = -fake_disc_out.mean()

            self.log(
                name="src_gen/loss",
                value=loss.cpu().item(),
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

            # adversarially train the src encoder with the src discriminator
            if self.src_discriminator is not None:
                assert self.enable_cycle_const
                real_repr = fake_repr_
                real_repr.register_hook(lambda grad: grad * self.grad_lambda)
                disc_out = self.src_discriminator(real_repr)
                disc_loss = disc_out.mean()
                loss += disc_loss

            if self.enable_cycle_const:
                cycle_loss = self.cyc_loss(
                    self.mapper_G(self.mapper_F(fake_repr_)), fake_repr_
                )
                cycle_loss = self.lambda_cc * cycle_loss

                loss += cycle_loss

                self.log(
                    name="src_enc/cyc_loss",
                    value=cycle_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

            return {"loss": loss}

        # target discriminator training
        if optimizer_idx == 4 and not self.disc_training_off:
            _, (real_repr, _) = self.trg_autoencoder.encode(
                trg_input_tensor,
                trg_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )
            _, (fake_repr, _) = self.src_autoencoder.encode(
                src_input_tensor,
                src_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )

            if self.enable_cycle_const:
                fake_repr = self.mapper_F(fake_repr)

            real_disc_out = self.trg_discriminator(real_repr.detach())
            fake_disc_out = self.trg_discriminator(fake_repr.detach())
            disc_loss = -(real_disc_out - fake_disc_out).mean()
            gradient_penalty = self.calc_gradient_penalty(
                real_repr.detach(), fake_repr.detach()
            )

            loss = disc_loss + gradient_penalty

            #########################
            # LOGGING
            #########################
            self.log(
                name="disc_trg/loss",
                value=loss.cpu().item(),
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=True,
            )

            return {
                "loss": loss,
            }

        if (
            (self.src_discriminator is not None)
            and (optimizer_idx == 5)
            and (not self.disc_training_off)
        ):
            _, (real_repr, _) = self.src_autoencoder.encode(
                src_input_tensor,
                src_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )
            _, (fake_repr, _) = self.trg_autoencoder.encode(
                trg_input_tensor,
                trg_input_len_tensor,
                noise=False,
                pad_idx=self.pad_idx,
            )

            if self.enable_cycle_const:
                fake_repr = self.mapper_G(fake_repr)

            real_disc_out = self.src_discriminator(real_repr.detach())
            fake_disc_out = self.src_discriminator(fake_repr.detach())
            disc_loss = -(real_disc_out - fake_disc_out).mean()
            gradient_penalty = self.calc_gradient_penalty(
                real_repr.detach(), fake_repr.detach()
            )

            loss = disc_loss + gradient_penalty

            #########################
            # LOGGING
            #########################
            self.log(
                name="disc_src/loss",
                value=loss.cpu().item(),
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=True,
            )

            return {
                "loss": loss,
            }

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

            if self.enable_cycle_const:
                self.optimizer_mapper_F.step()
                self.optimizer_mapper_F.zero_grad()
                self.optimizer_mapper_G.step()
                self.optimizer_mapper_G.zero_grad()

        # src encoder adversarial training
        if optimizer_idx == 3:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()

            if self.enable_cycle_const:
                self.optimizer_mapper_F.step()
                self.optimizer_mapper_F.zero_grad()
                self.optimizer_mapper_G.step()
                self.optimizer_mapper_G.zero_grad()

        # trg_discriminator training
        if optimizer_idx == 4:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()

        # src discriminator training
        if self.src_discriminator is not None and optimizer_idx == 5:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()

    def configure_optimizers(self):
        optim_src_ae = optim.Adam(
            self.src_autoencoder.parameters(),
            lr=self.lr_gen,
            betas=(0.5, 0.999),
        )

        if self.single_encoder_two_decoders or self.enable_cycle_const:
            optim_trg_ae = optim.Adam(self.trg_autoencoder.parameters(), lr=self.lr_ae)
        else:
            optim_trg_ae = optim.SGD(self.trg_autoencoder.parameters(), lr=self.lr_ae)
        optim_trg_disc = optim.Adam(
            self.trg_discriminator.parameters(),
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

        if self.src_discriminator is not None:
            optim_src_disc = optim.Adam(
                self.src_discriminator.parameters(),
                lr=self.lr_disc,
                betas=(0.5, 0.999),
            )
            optimizers.append(
                {
                    "optimizer": optim_src_disc,
                    "frequency": self.disc_num_steps,
                }
            )

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
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(
            True
        )
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

        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1) ** 2
        ).mean() * self.gan_gp_lambda
        return gradient_penalty

    def validation_step(self, batch, batch_idx, dataloader_idx):

        # source data loader
        (
            input_tensor,
            input_len_tensor,
            dec_inp_tensor,
            dec_output_tensor,
            mask_tensor,
            _
        ) = batch
        input_len_tensor = input_len_tensor.cpu()
        input_len_tensor = input_len_tensor.squeeze(1)  # B

        if dataloader_idx == 0:
            dev_sentences = tensors_to_text(
                tensors=input_tensor, idx2token=self.idx2word
            )

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

                # if cycle consistency is enabled map the representation to the target domain
                if self.enable_cycle_const:
                    fake_repr = self.mapper_F(fake_repr)

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
                {
                    "Dev Translations": wandb.Table(
                        data=table_data, columns=["Src", "Trg"]
                    )
                },
                commit=False,
            )

            # target sentences should be labelled with __label__2 in fast text
            # for the model
            # returns predictions, probabilities. Taking only predictions
            domain_predictions = self.ft_model.predict(generated_sentences)[0]
            # For every line the prediction can contain multiple predictions
            # Since ours is single prediction per line, take the first element
            domain_predictions = list(map(lambda pred: pred[0], domain_predictions))
            domain_predictions = [
                self.ft_style_mapping[pred] for pred in domain_predictions
            ]
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

        # instance wise metric calculation
        assert len(domain_predictions) == len(acceptable_preds) == len(sim_scores)

        ############################################
        # Calculate instance wise aggregate metrics
        ############################################
        aggregate = torch.mul(
            torch.Tensor(domain_predictions), torch.Tensor(acceptable_preds)
        )
        aggregate = torch.mul(aggregate, torch.Tensor(sim_scores))
        dev_aggregate = torch.mean(aggregate)

        self.log(
            name="dev_agg",
            value=dev_aggregate,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            name="src_enc/val_loss",
            value=avg_src_dev_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            name="trg_enc/val_loss",
            value=avg_trg_dev_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            name="dev_acceptability",
            value=percent_acceptable,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        self.log(
            name="dev_sim",
            value=avg_sim_score,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        self.log(
            name="dev_transfer_acc",
            value=avg_transfer_accs,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        generated_sents_file = self.generated_sents_dir.joinpath(
            f"{self.current_epoch}.txt"
        )
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
                _
            ) = batch
            test_sentences_ = tensors_to_text(
                tensors=input_tensor.cpu(), idx2token=self.idx2word
            )
            test_sentences.extend(test_sentences_)
            input_len_tensor = input_len_tensor.cpu().squeeze(1)  # B

            if self.generate_through_trg:
                (
                    encoder_outputs,
                    (fake_repr, encoder_cn),
                ) = self.trg_autoencoder.encode(
                    input_tensor,
                    input_len_tensor,
                    noise=False,
                    pad_idx=self.pad_idx,
                )
            else:
                (
                    encoder_outputs,
                    (fake_repr, encoder_cn),
                ) = self.src_autoencoder.encode(
                    input_tensor,
                    input_len_tensor,
                    noise=False,
                    pad_idx=self.pad_idx,
                )

                if self.enable_cycle_const:
                    fake_repr = self.mapper_F(fake_repr)

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
        domain_predictions = [
            self.ft_style_mapping[pred] for pred in domain_predictions
        ]
        domain_predictions = torch.LongTensor(domain_predictions)
        true_labels = torch.LongTensor([1] * len(domain_predictions))
        transfer_acc = self.test_transfer_accuracy(domain_predictions, true_labels)

        aggregate = torch.mul(
            domain_predictions.float(), torch.Tensor(acceptable_preds)
        )
        aggregate = torch.mul(aggregate, torch.Tensor(sim_scores))
        test_aggregate = torch.mean(aggregate)

        self.log(
            name="test_agg",
            value=test_aggregate,
            prog_bar=False,
            on_step=False,
            logger=True,
            on_epoch=True,
        )

        self.log(
            name="test_sim",
            value=avg_sim_score,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        self.log(
            name="test_transfer_acc",
            value=transfer_acc,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            name="test_percent_acceptable",
            value=percent_acceptable,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.logger.experiment.log(
            {
                "Test Translations": wandb.Table(
                    data=list(zip(test_sents, generated_sents)),
                    columns=["Src", "Trg"],
                )
            },
        )

        test_sents_from_filename = self.generated_sents_dir.joinpath("test.from")
        test_sents_to_filename = self.generated_sents_dir.joinpath("test.to")

        with open(test_sents_from_filename, "w") as fp:
            for line in test_sents:
                fp.write(line)
                fp.write("\n")

        with open(test_sents_to_filename, "w") as fp:
            for line in generated_sents:
                fp.write(line)
                fp.write("\n")
