import click
import pathlib
from dct.data.pl_datamodule import DataModule
from dct.pl_arae_contra import PLARAE
from dct.models.autoencoder import Seq2Seq
from dct.models.discriminator import Critic
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
import torch.nn as nn
import fasttext
import json
from dct.console import console
import shutil
from rich.prompt import Confirm


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
    "--src-train-label-file",
    type=str,
    required=False,
    help="Training label file for src domain",
)
@click.option(
    "--trg-train-label-file",
    type=str,
    required=False,
    help="Training label file for trg domain",
)
@click.option(
    "--src-dev-label-file",
    type=str,
    required=False,
    help="Dev label file for src domain",
)
@click.option(
    "--trg-dev-label-file",
    type=str,
    required=False,
    help="Training label file for trg domain",
)
@click.option(
    "--src-test-label-file",
    type=str,
    required=False,
    help="Test label file for src domain",
)
@click.option(
    "--trg-test-label-file",
    type=str,
    required=False,
    help="Testing label file for trg domain",
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
    "--max_vocab_size",
    type=int,
    required=True,
    help="Maximum Vocab Size to consider",
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
    "--log_ae_every_iter",
    type=int,
    required=False,
    default=50,
    help="Log the reconstructions of the autoencoder every",
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
@click.option("--single-encoder-two-decoders", is_flag=True)
@click.option(
    "--tie-encoder-embeddings",
    is_flag=True,
    help="Ties the encoder embeddings of the source and the target encoder",
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
@click.option(
    "--encoder-contrastive-learning-off",
    is_flag=True,
    help="Disables constrastive learning part of the model",
)
@click.option(
    "--discriminator-contrastive-learning-off",
    is_flag=True,
    help="Switches off contrastive learning for the encoder",
)
@click.option(
    "--num-nearest-labels",
    type=int,
    help="Number of nearest labels to consider while obtaining positives "
    "for contrastive learning",
    required=True,
    default=20,
)
@click.option(
    "--normalize-contrastive-tensors",
    is_flag=True,
    help="If set, takes the average of contrastive loss, instead of sum",
)
@click.option(
    "--contrastive-loss-reduction",
    type=str,
    help="How to reduce the contrastive loss? sum or mean",
    default="sum",
)
@click.option(
    "--use-official-sup-con-loss",
    is_flag=True,
    help="Uses official supervised contrastive loss",
)
@click.option("--weight-contrastive-loss", type=float, required=False, default=1.0)
@click.option(
    "--generator-only-contrastive-learning",
    is_flag=True,
    help="If True, we add contrastive learning to the generative part of the encode only",
)
@click.option(
    "--use-disc-attr-clf",
    is_flag=True,
    help="CLF baseline on the discriminator",
)
@click.option(
    "--use-enc-attr-clf",
    is_flag=True,
    help="CLF baseline on the discriminator",
)
@click.option(
    "--arch-contra-clf-baseline",
    type=str,
    required=False,
    default=None,
    help="Architecture for the contrastive",
)
@click.option(
    "--src-saliency-file", type=str, required=True, help="Src domain saliency file"
)
@click.option(
    "--trg-saliency-file", type=str, required=True, help="Trg domain saliency file"
)
@click.option(
    "--wandb-proj-name",
    type=str,
    required=False,
    default="contrarae",
    help="Project name in the wandb work space",
)
@click.option(
    "--vocab_file",
    type=str,
    required=False,
    default=None,
    help="If vocab file is provided, then we will use that."
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
    src_train_label_file,
    trg_train_label_file,
    src_dev_label_file,
    trg_dev_label_file,
    src_test_label_file,
    trg_test_label_file,
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
    log_ae_every_iter,
    arch_d,
    arch_g,
    z_size,
    gan_gp_lambda,
    autoencoder_training_off,
    disc_training_off,
    adv_training_off,
    grad_lambda,
    seed,
    no_early_stopping,
    no_checkpointing,
    generate_through_trg,
    precision,
    single_encoder_two_decoders,
    tie_encoder_embeddings,
    gen_greedy,
    ft_model_path,
    lambda_ae,
    lambda_cc,
    lambda_adv,
    no_disc_batchnorm,
    cola_roberta_checkpoints_dir,
    cola_roberta_json_file,
    sim_model,
    sim_sentencepiece_model,
    encoder_contrastive_learning_off,
    discriminator_contrastive_learning_off,
    num_nearest_labels,
    normalize_contrastive_tensors,
    contrastive_loss_reduction,
    use_official_sup_con_loss,
    weight_contrastive_loss,
    generator_only_contrastive_learning,
    use_disc_attr_clf,
    use_enc_attr_clf,
    arch_contra_clf_baseline,
    src_saliency_file,
    trg_saliency_file,
    wandb_proj_name,
    vocab_file
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
        "src_train_label_file": src_train_label_file,
        "trg_train_label_file": trg_train_label_file,
        "src_dev_label_file": src_dev_label_file,
        "trg_dev_label_file": trg_dev_label_file,
        "src_test_label_file": src_test_label_file,
        "trg_test_label_file": trg_test_label_file,
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
        "log_ae_every_iter": log_ae_every_iter,
        "arch_d": arch_d,
        "arch_g": arch_g,
        "z_size": z_size,
        "gan_gp_lambda": gan_gp_lambda,
        "autoencoder_training_off": autoencoder_training_off,
        "disc_training_off": disc_training_off,
        "adv_training_off": adv_training_off,
        "grad_lambda": grad_lambda,
        "seed": seed,
        "generate_through_trg": generate_through_trg,
        "single_encoder_two_decoders": single_encoder_two_decoders,
        "tie_encoder_embeddings": tie_encoder_embeddings,
        "gen_greedy": gen_greedy,
        "ft_model_path": ft_model_path,
        "lambda_ae": lambda_ae,
        "lambda_cc": lambda_cc,
        "lambda_adv": lambda_adv,
        "no_disc_batchnorm": no_disc_batchnorm,
        "cola_roberta_checkpoints_dir": cola_roberta_checkpoints_dir,
        "cola_roberta_json_file": cola_roberta_json_file,
        "sim_model": sim_model,
        "sim_sentencepiece_model": sim_sentencepiece_model,
        "encoder_contrastive_learning_off": encoder_contrastive_learning_off,
        "discriminator_contrastive_learning_off": discriminator_contrastive_learning_off,
        "num_nearest_labels": num_nearest_labels,
        "normalize_contrastive_tensors": normalize_contrastive_tensors,
        "contrastive_loss_reduction": contrastive_loss_reduction,
        "use_official_sup_con_loss": use_official_sup_con_loss,
        "weight_contrastive_loss": weight_contrastive_loss,
        "generator_only_contrastive_learning": generator_only_contrastive_learning,
        "use_disc_attr_clf": use_disc_attr_clf,
        "use_enc_attr_clf": use_enc_attr_clf,
        "arch_contra_clf_baseline": arch_contra_clf_baseline,
        "src_saliency_file": src_saliency_file,
        "trg_saliency_file": trg_saliency_file,
        "wandb_proj_name": wandb_proj_name,
        "vocab_file": vocab_file
    }

    cola_roberta_json_file = pathlib.Path(cola_roberta_json_file)
    cola_roberta_checkpoints_dir = pathlib.Path(cola_roberta_checkpoints_dir)
    ###########################################################################
    # SETUP DATASETS
    ###########################################################################
    if vocab_file is None:
        vocab_file = exp_dir.joinpath("vocab.txt")
    else:
        vocab_file = pathlib.Path(vocab_file)
    label_vocab_file = exp_dir.joinpath("label_vocab.txt")

    opts["vocab_file"] = str(vocab_file)
    opts["label_vocab_file"] = str(label_vocab_file)

    dm = DataModule(opts)
    dm.prepare_data()
    vocab_size = dm.get_vocab_size()
    label_vocab_size = dm.get_label_vocab_size()

    opts["vocab_size"] = vocab_size
    opts["label_vocab_size"] = label_vocab_size
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

    if single_encoder_two_decoders or tie_encoder_embeddings:
        src_autoencoder.enc_embedding.weight = trg_autoencoder.enc_embedding.weight

    trg_discriminator = Critic.from_opts(opts)

    if use_disc_attr_clf or use_enc_attr_clf:
        layer_sizes = list(map(int, arch_contra_clf_baseline.split("-")))
        contra_baseline_clf = Critic(
            ninput=opts["ae_hidden_size"],
            noutput=label_vocab_size,
            layer_sizes=layer_sizes,
        )
    else:
        contra_baseline_clf = None

    # Load the fasttext model
    ft_model = fasttext.load_model(ft_model_path)
    ###########################################################################
    # SETUP TRAINER
    ###########################################################################
    model = PLARAE(
        src_autoencoder=src_autoencoder,
        trg_autoencoder=trg_autoencoder,
        trg_discriminator=trg_discriminator,
        datamodule=dm,
        ft_model=ft_model,
        hparams=opts,
        cola_roberta_checkpoints_dir=cola_roberta_checkpoints_dir,
        cola_roberta_json_file=cola_roberta_json_file,
        contrastive_baseline_classifier=contra_baseline_clf,
        src_saliency_file=src_saliency_file,
        trg_saliency_file=trg_saliency_file,
    )

    ####################################
    # SETUP LOGGERS
    ####################################
    exp_dir = pathlib.Path(opts["exp_dir"])
    logger = WandbLogger(
        name=opts["exp_name"],
        save_dir=str(exp_dir),
        project=wandb_proj_name,
    )

    logger.watch(model, log="gradients", log_freq=10)

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
    trainer.test()

    hparams_file = exp_dir.joinpath("hparams.json")

    opts.pop("encoder")
    with open(hparams_file, "w") as fp:
        json.dump(opts, fp)


if __name__ == "__main__":
    train()
