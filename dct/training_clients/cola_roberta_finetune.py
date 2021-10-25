import click
from dct.data.cola_datamodule import ColaDataModule
from transformers import AutoTokenizer
from dct.models.roberta_finetune import RobertaFineTune
from pytorch_lightning import Trainer
import pathlib
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
import json


@click.command()
@click.option("--exp-dir", type=str, help="Directory in which the experiment is stored")
@click.option(
    "--exp-name", type=str, help="Directory in which the experiment is stored"
)
@click.option(
    "--gpu", type=int, required=False, default=-1, help="The GPU to use if any else -1"
)
@click.option(
    "--encoder-model",
    default="roberta-large",
    type=str,
    help="Encoder model to be used.",
)
@click.option(
    "--encoder-lr",
    default=1e-05,
    type=float,
    help="Encoder specific learning rate.",
)
@click.option(
    "--clf-lr",
    default=3e-05,
    type=float,
    help="Classification head learning rate.",
)
@click.option(
    "--nr-frozen-epochs",
    default=1,
    type=int,
    help="Number of epochs we want to keep the encoder model frozen.",
)
@click.option(
    "--loader-workers",
    default=8,
    type=int,
    help="How many subprocesses to use for data loading. 0 means that \
           the data will be loaded in the main process.",
)
@click.option(
    "--batch-size",
    default=32,
    type=int,
    help="How many subprocesses to use for data loading. 0 means that \
           the data will be loaded in the main process.",
)
@click.option(
    "--accumulate-grad-batches", type=int, help="Accumulate gradients every k batches"
)
@click.option("--epochs", type=int, help="Number of epochs")
@click.option(
    "--limit-train-proportion",
    type=float,
    required=False,
    default=1.0,
    help="Limits the training data to a percentage",
)
@click.option(
    "--limit-dev-proportion",
    type=float,
    required=False,
    default=1.0,
    help="Limits the training data to a percentage",
)
@click.option("--seed", type=int, required=False, default=1729, help="Seed everything")
def cola_roberta_finetune(
    exp_dir,
    exp_name,
    gpu,
    encoder_model,
    encoder_lr,
    clf_lr,
    nr_frozen_epochs,
    loader_workers,
    batch_size,
    accumulate_grad_batches,
    epochs,
    limit_train_proportion,
    limit_dev_proportion,
    seed,
):

    hparams = {
        "exp_dir": exp_dir,
        "exp_name": exp_name,
        "gpu": gpu,
        "encoder_model": encoder_model,
        "encoder_lr": encoder_lr,
        "clf_lr": clf_lr,
        "batch_size": batch_size,
        "nr_frozen_epochs": nr_frozen_epochs,
        "num_workers": loader_workers,
        "accumulate_grad_batches": accumulate_grad_batches,
        "epochs": epochs,
        "limit_train_proportion": limit_train_proportion,
        "limit_dev_proportion": limit_dev_proportion,
        "seed": seed,
    }

    exp_dir = pathlib.Path(exp_dir)

    if not exp_dir.is_dir():
        exp_dir.mkdir(parents=True)

    with open(exp_dir.joinpath(f"hparams.json"), "w") as fp:
        json.dump(hparams, fp)

    seed_everything(hparams["seed"])
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    cola_dm = ColaDataModule(
        {"batch_size": batch_size, "num_workers": loader_workers}, tokenizer=tokenizer
    )
    model = RobertaFineTune(hparams=hparams, datamodule=cola_dm)

    exp_dir = pathlib.Path(hparams["exp_dir"])
    logger = WandbLogger(
        name=hparams["exp_name"],
        save_dir=str(exp_dir),
        project="robertabase-cola",
    )

    callbacks = []

    early_stopping = EarlyStopping(
        monitor="dev_acc", mode="max", verbose=True, patience=10
    )

    callbacks.append(early_stopping)

    checkpoints_dir = exp_dir.joinpath("checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir), save_top_k=1, mode="max", monitor="dev_acc"
    )
    callbacks.append(checkpoint_callback)

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        gpus=str(hparams["gpu"]),
        deterministic=True,
        fast_dev_run=False,
        accumulate_grad_batches=int(hparams["accumulate_grad_batches"]),
        max_epochs=hparams["epochs"],
        limit_train_batches=hparams["limit_train_proportion"],
        limit_val_batches=hparams["limit_dev_proportion"],
        terminate_on_nan=True,
    )
    # ------------------------
    # 6 START TRAINING
    # ------------------------
    cola_dm.setup("fit")
    trainer.fit(model, datamodule=cola_dm)

    cola_dm.setup("test")
    trainer.test()


if __name__ == "__main__":
    cola_roberta_finetune()
