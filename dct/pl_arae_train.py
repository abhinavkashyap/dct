import click
import pathlib
from dct.data.pl_datamodule import DataModule
from dct.pl_arae import PLARAE
from dct.models.autoencoder import Seq2Seq
from dct.models.discriminator import Critic
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
from dct.models.cycle_mapper import CycleMapper
import torch.nn as nn
import fasttext
import json
from dct.console import console
from rich.prompt import Confirm
import shutil

@click.command()
@click.option("--exp_name", type=str, required=True, help="Experiment Name")
@click.option("--exp_dir", type=str, required=True, help="Experiment directory")
@click.option(
    "--src_train_file",
    type=str,
    required=True,
    help="Train file for the source domain",
)
@click.option(
    "--trg_train_file",
    type=str,
    required=True,
    help="Train file for the target domain",
)
@click.option(
    "--src_dev_file",
    type=str,
    required=True,
    help="Dev file for the source domain",
)
@click.option(
    "--trg_dev_file",
    type=str,
    required=True,
    help="Dev file for the target domain",
)
@click.option(
    "--src_test_file",
    type=str,
    required=True,
    help="Test file for the source domain",
)
@click.option(
    "--trg_test_file",
    type=str,
    required=True,
    help="Test file for the target domain",
)
@click.option(
    "--max_seq_length",
    type=int,
    required=True,
    help="Maximum sequence length to consider",
)
@click.option("--batch_size", type=int, required=True, help="Batch size")
@click.option(
    "--num_processes",
    type=int,
    required=True,
    help="Number of processors to use for data loading",
)
@click.option(
    "--limit_train_proportion",
    type=float,
    required=False,
    default=1.0,
    help="Limits the training data to a percentage",
)
@click.option(
    "--limit_dev_proportion",
    type=float,
    required=False,
    default=1.0,
    help="Limits the training data to a percentage",
)
@click.option(
    "--max_vocab_size", type=int, help="Maximum Vocab Size to consider", default=None
)
@click.option(
    "--gpu",
    type=int,
    required=False,
    default=-1,
    help="The GPU to use if any else -1",
)
@click.option(
    "--ae_emb_size",
    type=int,
    required=True,
    help="Embedding size of autoencoder",
)
@click.option(
    "--ae_hidden_size",
    type=int,
    required=True,
    help="LSTM hidden size of autoencoder",
)
@click.option(
    "--ae_n_layers",
    type=int,
    required=False,
    default=1,
    help="Number of LSTM layers for auetoencoder",
)
@click.option(
    "--noise_r", type=float, default=0.05, help="std of noise for autoencoder"
)
@click.option(
    "--ae_dropout",
    type=float,
    default=0,
    help="Dropout between multiple layer LSTM autoencoder",
)
@click.option("--lr_ae", type=float, default=1.0, help="Learning rate for autoencoder")
@click.option(
    "--lr-disc", type=float, default=1e-4, help="Learning rate of discriminator"
)
@click.option("--lr-gen", type=float, default=1e-04, help="Learning rate for geneartor")
@click.option("--disc_num_steps", type=int, default=5, help="Disc Numb steps")
@click.option(
    "--ae_anneal_noise_every",
    type=float,
    default=100,
    help="Noise added in autoencoder is annealed every few iterations",
)
@click.option("--ae_anneal_noise", type=float, default=0.9995, help="Annealing Noise")
@click.option(
    "--n_epochs",
    type=int,
    required=False,
    default=15,
    help="Number of epochs to train",
)
@click.option(
    "--grad_clip_norm",
    type=float,
    required=False,
    default=1.0,
    help="If the gradient norm is more than this, it will be clipped",
)
@click.option(
    "--arch_d",
    type=str,
    default="300-300",
    help="critic/discriminator architecture (MLP)",
)
@click.option(
    "--arch_g", type=str, default="300-300", help="generator architecture (MLP)"
)
@click.option(
    "--arch_mapper",
    type=str,
    default="300-300",
    help="generator architecture (MLP)",
)
@click.option(
    "--z_size",
    type=int,
    default=100,
    help="dimension of random noise z to feed into generator",
)
@click.option("--gan_gp_lambda", type=float, default=1, help="WGAN GP penalty lambda")
@click.option("--autoencoder-training-off", is_flag=True)
@click.option("--disc-training-off", is_flag=True)
@click.option("--adv-training-off", is_flag=True)
@click.option(
    "--grad_lambda",
    type=float,
    default=0.1,
    help="Gradient Lambda when training encoder adversarially",
)
@click.option(
    "--devoff",
    is_flag=True,
    help="If set, then development epochs will be skipped",
)
@click.option(
    "--seed",
    type=int,
    required=False,
    default=1729,
    help="Seed for reproducibility",
)
@click.option("--no-early-stopping", is_flag=True)
@click.option("--no-checkpointing", is_flag=True)
@click.option("--generate-through-trg", is_flag=True)
@click.option(
    "--precision",
    required=False,
    type=int,
    default=32,
    help="You can train using half precision",
)
@click.option("--enable-cycle-consistency", is_flag=True)
@click.option("--cycle-mapper-only-linear", is_flag=True)
@click.option(
    "--mapper-lr",
    required=False,
    default=1e-03,
    help="Learning rate for cycle consistency mappers",
)
@click.option("--single-encoder-two-decoders", is_flag=True)
@click.option(
    "--tie-encoder-embeddings",
    is_flag=True,
    help="Ties the encoder embeddings of the source and the target encoder",
)
@click.option(
    "--tie-enc-dec-embedding",
    is_flag=True,
    help="Use the same embedding matrix for the encoder and the decoder",
)
@click.option(
    "--add-src-side-disc",
    is_flag=True,
    help="If cycle consistency is used, you can use this flag to add another discriminator on the source side",
)
@click.option(
    "--gen-greedy",
    is_flag=True,
    help="By default we use the beam search. This can be set to generate using greedy decoding method",
)
@click.option(
    "--ft-model-path",
    type=str,
    required=True,
    help="FastText model that is used to record transfer accuracy during transfer",
)
@click.option(
    "--lambda-ae",
    type=float,
    required=False,
    default=1,
    help="Autoencoder Lambda.. Used to weigh the autoencoder loss",
)
@click.option(
    "--lambda-cc",
    type=float,
    required=False,
    default=1,
    help="Cycle Consistency Lambda.. Used to weigh the cycle consistency loss",
)
@click.option(
    "--lambda-adv",
    type=float,
    required=True,
    default=1,
    help="Adversarial loss Lambda.. Used to weigh the adversarial loss",
)
@click.option("--no-disc-batchnorm", is_flag=True)
@click.option("--train-mapper-detached", is_flag=True)
@click.option(
    "--cola-roberta-checkpoints-dir",
    type=str,
    help="Checkpoints directory containing roberta finetuned on cola",
)
@click.option(
    "--cola-roberta-json-file",
    type=str,
    help="Hyperparams used for training roberta on cola dataset",
)
@click.option("--sim-model", type=str, help="Semantic Similarity Model")
@click.option(
    "--sim-sentencepiece-model",
    type=str,
    help="Semantic Similarity Sentence Piece Model",
)
def train(
    exp_name,
    exp_dir,
    src_train_file,
    trg_train_file,
    src_dev_file,
    trg_dev_file,
    src_test_file,
    trg_test_file,
    max_seq_length,
    batch_size,
    num_processes,
    limit_train_proportion,
    limit_dev_proportion,
    max_vocab_size,
    gpu,
    ae_emb_size,
    ae_hidden_size,
    ae_n_layers,
    noise_r,
    ae_dropout,
    lr_ae,
    lr_disc,
    lr_gen,
    disc_num_steps,
    ae_anneal_noise_every,
    ae_anneal_noise,
    n_epochs,
    grad_clip_norm,
    arch_d,
    arch_g,
    arch_mapper,
    z_size,
    gan_gp_lambda,
    autoencoder_training_off,
    disc_training_off,
    adv_training_off,
    grad_lambda,
    devoff,
    seed,
    no_early_stopping,
    no_checkpointing,
    generate_through_trg,
    precision,
    enable_cycle_consistency,
    cycle_mapper_only_linear,
    mapper_lr,
    single_encoder_two_decoders,
    tie_encoder_embeddings,
    tie_enc_dec_embedding,
    add_src_side_disc,
    gen_greedy,
    ft_model_path,
    lambda_ae,
    lambda_cc,
    lambda_adv,
    no_disc_batchnorm,
    train_mapper_detached,
    cola_roberta_checkpoints_dir,
    cola_roberta_json_file,
    sim_model,
    sim_sentencepiece_model,
):
    exp_dir = pathlib.Path(exp_dir)
    if not exp_dir.is_dir():
        exp_dir.mkdir(parents=True)
    else:
        is_delete = Confirm.ask(f"{exp_dir} already exists. Delete it?")
        if is_delete:
            shutil.rmtree(exp_dir)
            console.print(f"[green] deleted {exp_dir}")
        exp_dir.mkdir(parents=True)

    seed_everything(seed)

    opts = {
        "exp_name": exp_name,
        "exp_dir": str(exp_dir),
        "src_train_file": src_train_file,
        "trg_train_file": trg_train_file,
        "src_dev_file": src_dev_file,
        "trg_dev_file": trg_dev_file,
        "src_test_file": src_test_file,
        "trg_test_file": trg_test_file,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "num_processes": num_processes,
        "limit_train_proportion": limit_train_proportion,
        "limit_dev_proportion": limit_dev_proportion,
        "max_vocab_size": max_vocab_size,
        "gpu": gpu,
        "ae_emb_size": ae_emb_size,
        "ae_hidden_size": ae_hidden_size,
        "ae_n_layers": ae_n_layers,
        "noise_r": noise_r,
        "ae_dropout": ae_dropout,
        "lr_ae": lr_ae,
        "lr_disc": lr_disc,
        "lr_gen": lr_gen,
        "disc_num_steps": disc_num_steps,
        "ae_anneal_noise_every": ae_anneal_noise_every,
        "ae_anneal_noise": ae_anneal_noise,
        "n_epochs": n_epochs,
        "grad_clip_norm": grad_clip_norm,
        "arch_d": arch_d,
        "arch_g": arch_g,
        "arch_mapper": arch_mapper,
        "z_size": z_size,
        "gan_gp_lambda": gan_gp_lambda,
        "autoencoder_training_off": autoencoder_training_off,
        "disc_training_off": disc_training_off,
        "adv_training_off": adv_training_off,
        "grad_lambda": grad_lambda,
        "devoff": devoff,
        "generate_through_trg": generate_through_trg,
        "enable_cycle_consistency": enable_cycle_consistency,
        "cycle_mapper_only_linear": cycle_mapper_only_linear,
        "mapper_lr": mapper_lr,
        "single_encoder_two_decoders": single_encoder_two_decoders,
        "gen_greedy": gen_greedy,
        "add_src_side_disc": add_src_side_disc,
        "lambda_ae": lambda_ae,
        "lambda_cc": lambda_cc,
        "lambda_adv": lambda_adv,
        "no_disc_batchnorm": no_disc_batchnorm,
        "train_mapper_detached": train_mapper_detached,
        "cola_roberta_checkpoints_dir": cola_roberta_checkpoints_dir,
        "cola_roberta_json_file": cola_roberta_json_file,
        "sim_model": sim_model,
        "sim_sentencepiece_model": sim_sentencepiece_model,
        "tie_enc_dec_embedding": tie_enc_dec_embedding,
        "tie_encoder_embeddings": tie_encoder_embeddings,
        "ft_model_path": ft_model_path,
        "seed": seed
    }

    cola_roberta_json_file = pathlib.Path(cola_roberta_json_file)
    cola_roberta_checkpoints_dir = pathlib.Path(cola_roberta_checkpoints_dir)
    ###########################################################################
    # SETUP DATASETS
    ###########################################################################
    vocab_file = exp_dir.joinpath("vocab.txt")

    opts["vocab_file"] = str(vocab_file)

    dm = DataModule(opts)
    dm.prepare_data()
    vocab_size = dm.get_vocab_size()

    opts["vocab_size"] = vocab_size
    ###########################################################################
    # SETUP MODELS
    ###########################################################################
    # for transfer from src to trg
    # src_autoencoder = fake
    # trg_autoencoder = real
    if single_encoder_two_decoders:
        encoder = nn.LSTM(
            input_size=opts["ae_emb_size"],
            hidden_size=opts["ae_hidden_size"],
            num_layers=1,
            dropout=opts["ae_dropout"],
            batch_first=True,
        )
        opts["encoder"] = encoder
    else:
        opts["encoder"] = None

    src_autoencoder = Seq2Seq.from_opts(opts)
    trg_autoencoder = Seq2Seq.from_opts(opts)

    # remove opts["encoder] from the dictionary because they are not serializable
    opts.pop("encoder")

    if single_encoder_two_decoders or tie_encoder_embeddings:
        src_autoencoder.enc_embedding.weight = trg_autoencoder.enc_embedding.weight

    trg_discriminator = Critic.from_opts(opts)

    if add_src_side_disc:
        src_discriminator = Critic.from_opts(opts)
    else:
        src_discriminator = None

    if opts["enable_cycle_consistency"]:
        mapper_F = CycleMapper.from_opts(opts)
        mapper_G = CycleMapper.from_opts(opts)

    else:
        mapper_F = None
        mapper_G = None

    # Load the fasttext model
    ft_model = fasttext.load_model(ft_model_path)
    ###########################################################################
    # SETUP TRAINER
    ###########################################################################
    model = PLARAE(
        src_autoencoder=src_autoencoder,
        trg_autoencoder=trg_autoencoder,
        trg_discriminator=trg_discriminator,
        src_discriminator=src_discriminator,
        mapper_F=mapper_F,
        mapper_G=mapper_G,
        datamodule=dm,
        ft_model=ft_model,
        hparams=opts,
        cola_roberta_checkpoints_dir=cola_roberta_checkpoints_dir,
        cola_roberta_json_file=cola_roberta_json_file,
    )

    ####################################
    # SETUP LOGGERS
    ####################################
    exp_dir = pathlib.Path(opts["exp_dir"])
    logger = WandbLogger(
        name=opts["exp_name"],
        save_dir=str(exp_dir),
        project="DCT",
    )

    logger.watch(model, log="gradients", log_freq=100)

    callbacks = []

    if not no_checkpointing:
        checkpoints_dir = exp_dir.joinpath("checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoints_dir),
            save_top_k=1,
            mode="max",
            monitor="dev_agg",
        )
        callbacks.append(checkpoint_callback)

    if not no_early_stopping:
        early_stopping = EarlyStopping(
            monitor="dev_agg", mode="max", verbose=True, patience=10
        )
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        limit_train_batches=opts["limit_train_proportion"],
        limit_val_batches=opts["limit_dev_proportion"],
        limit_test_batches=opts["limit_dev_proportion"],
        max_epochs=opts["n_epochs"],
        logger=logger,
        gpus=str(opts["gpu"]),
        gradient_clip_val=opts["grad_clip_norm"],
        callbacks=callbacks,
        terminate_on_nan=True,
        precision=precision,
    )
    dm.setup("fit")
    trainer.fit(model)

    # Do not pass the model here
    # Loads the best model from checkpoint directory and then tests it
    dm.setup("test")
    trainer.test(ckpt_path="best")

    hparams_file = exp_dir.joinpath("hparams.json")

    with open(hparams_file, "w") as fp:
        json.dump(opts, fp)


if __name__ == "__main__":
    train()
