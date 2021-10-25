import pathlib
import wasabi
import torch
from dct.pl_arae import PLARAE
from typing import List
from dct.data.pl_datamodule import DataModule
from dct.data.pl_dataset import PLDataset
import torch.nn as nn
from dct.models.autoencoder import Seq2Seq
from dct.models.discriminator import Critic
from dct.models.cycle_mapper import CycleMapper
import fasttext
import json
from torch.utils.data import DataLoader
import multiprocessing
import numpy as np
from tqdm import tqdm
from dct.utils.tensor_utils import tensors_to_text
from dct.console import console


class ARAEInfer:
    def __init__(self, checkpoints_dir: pathlib.Path, hparams_file: pathlib.Path):
        """ Recreate the model from the checkpoints directory and the
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
        self.msg_printer = wasabi.Printer()

        if not self.hparams_file.is_file():
            console.print(f"[red] {self.hparams_file} does not exist... Loading Model Failed")
            exit(1)

        with open(str(self.hparams_file), "r") as fp:
            self.hparams = json.load(fp)

        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.single_encoder_two_decoders = self.hparams["single_encoder_two_decoders"]
        self.tie_encoder_embeddings = self.hparams["tie_encoder_embeddings"]
        self.add_src_side_disc = self.hparams["add_src_side_disc"]
        self.enable_cycle_consistency = self.hparams["enable_cycle_consistency"]
        self.ft_model_path = self.hparams["ft_model_path"]
        self.cola_ft_model_path = None
        self.cola_roberta_checkpoints_dir = self.hparams["cola_roberta_checkpoints_dir"]
        self.cola_roberta_json_file = self.hparams["cola_roberta_json_file"]
        self.gpu = self.hparams.get("gpu", -1)
        self.max_seq_length = self.hparams.get("max_seq_length", 50)
        self.batch_size = self.hparams.get("batch_size", 32)
        self.generate_through_trg = self.hparams.get("generate_through_trg", True)
        self.enable_cycle_const = self.hparams.get("enable_cycle_consistency")
        self.datamodule = DataModule(self.hparams)
        self.idx2word = self.datamodule.vocab.get_idx2word()
        self.word2idx = self.datamodule.vocab.get_word2idx()
        self.pad_idx = self.word2idx["<pad>"]
        self.sos_idx = self.word2idx["<s>"]
        self.eos_idx = self.word2idx["</s>"]
        self.model = self._load_model()

    def _load_model(self):
        filenames = list(self.checkpoints_dir.iterdir())
        assert (
            len(filenames) == 1
        ), f"Make sure only the best model is stored in the checkpoints directory"
        best_filename = self.checkpoints_dir.joinpath(filenames[-1])

        with self.msg_printer.loading(f"Loading Model"):

            if self.single_encoder_two_decoders:
                encoder = nn.LSTM(
                    input_size=self.hparams["ae_emb_size"],
                    hidden_size=self.hparams["ae_hidden_size"],
                    num_layers=1,
                    dropout=self.hparams["ae_dropout"],
                    batch_first=True,
                )
                self.hparams["encoder"] = encoder
            else:
                self.hparams["encoder"] = None

            src_autoencoder = Seq2Seq.from_opts(self.hparams)
            trg_autoencoder = Seq2Seq.from_opts(self.hparams)

            if self.single_encoder_two_decoders or self.tie_encoder_embeddings:
                src_autoencoder.enc_embedding.weight = trg_autoencoder.enc_embedding.weight

            trg_discriminator = Critic.from_opts(self.hparams)

            if self.add_src_side_disc:
                src_discriminator = Critic.from_opts(self.hparams)
            else:
                src_discriminator = None

            if self.enable_cycle_consistency:
                mapper_F = CycleMapper.from_opts(self.hparams)
                mapper_G = CycleMapper.from_opts(self.hparams)

            else:
                mapper_F = None
                mapper_G = None

            ft_model = fasttext.load_model(self.ft_model_path)

            model = PLARAE.load_from_checkpoint(
                str(best_filename),
                hparams=self.hparams,
                src_autoencoder=src_autoencoder,
                trg_autoencoder=trg_autoencoder,
                trg_discriminator=trg_discriminator,
                mapper_F=mapper_F,
                mapper_G=mapper_G,
                src_discriminator=src_discriminator,
                datamodule=self.datamodule,
                ft_model=ft_model,
                cola_ft_model=None,
                cola_roberta_checkpoints_dir=self.cola_roberta_checkpoints_dir,
                cola_roberta_json_file=self.cola_roberta_json_file,
            )
            if torch.cuda.is_available():
                model.to(self.device)

        self.msg_printer.good(f"Finished Loading Model from {best_filename}")
        return model

    def predict(
        self,
        lines: List[str],
        gen_greedy=True,
        nucleus_sampling=True,
        top_k=0,
        top_p: float = 1.0,
        min_tokens_to_keep=1,
        temperature=1.0
    ):
        dataset = PLDataset(
            lines=lines,
            vocabulary=self.datamodule.vocab,
            stage="test",
            max_length=self.max_seq_length,
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

        test_sentences = []
        generated_sentences = []
        for batch in tqdm(loader, total=total_batches, desc="ARAE infer"):
            (
                input_tensor,
                input_len_tensor,
                dec_inp_tensor,
                dec_output_tensor,
                _,
                _,
            ) = batch
            test_sentences_ = tensors_to_text(tensors=input_tensor.cpu(), idx2token=self.idx2word)
            test_sentences.extend(test_sentences_)
            input_len_tensor = input_len_tensor.cpu().squeeze(1)  # B

            input_tensor = input_tensor.to(self.device)

            if self.generate_through_trg:
                (encoder_outputs, (fake_repr, encoder_cn),) = self.model.trg_autoencoder.encode(
                    input_tensor,
                    input_len_tensor,
                    noise=False,
                    pad_idx=self.pad_idx,
                )
            else:
                (encoder_outputs, (fake_repr, encoder_cn),) = self.model.src_autoencoder.encode(
                    input_tensor,
                    input_len_tensor,
                    noise=False,
                    pad_idx=self.pad_idx,
                )

                if self.enable_cycle_const:
                    fake_repr = self.model.mapper_F(fake_repr)

            generated_idxs = self.model.trg_autoencoder.generate(
                encoder_outputs,
                (fake_repr, encoder_cn),
                fake_repr,
                self.sos_idx,
                self.eos_idx,
                self.max_seq_length,
                gen_greedy=gen_greedy,
                nucleus_sampling=nucleus_sampling,
                top_k=top_k,
                top_p=top_p,
                min_tokens_to_keep=1,
                temperature=temperature
            )

            generated_sentences_ = tensors_to_text(generated_idxs, self.idx2word)
            generated_sentences.extend(generated_sentences_)
        return generated_sentences


if __name__ == "__main__":
    experiments_dir = pathlib.Path(
        "/data3/abhinav/dct/experiments/[TEST]"
    )
    checkpoints_dir = experiments_dir.joinpath("checkpoints")
    hparams_file = experiments_dir.joinpath("hparams.json")

    imdb_data_dir = pathlib.Path("/home/rkashyap/abhi/dct/data/yelp")
    test_from_file = imdb_data_dir.joinpath("sentiment.test.0")

    infer = ARAEInfer(checkpoints_dir=checkpoints_dir, hparams_file=hparams_file)

    with open(test_from_file, "r") as fp:
        lines = [line.strip() for line in fp]
        transferred_sentences = infer.predict(
            lines=lines,
            nucleus_sampling=True,
            gen_greedy=False,
            top_p=0.6,
        )

    with open("imdb_gen_ns_0.6.txt", "w") as fp:
        for src_sent, gen_sent in zip(lines, transferred_sentences):
            fp.write(src_sent.strip())
            fp.write("\n")
            fp.write(gen_sent)
            fp.write("\n")
            fp.write("========")
            fp.write("\n")

