import click
from dct.data.dann_datamodule import DANNDataModule
from dct.console import console
from rich.prompt import Confirm
import pathlib
import shutil
from dct.models.dann import DANN
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import json
from pytorch_lightning import seed_everything


@click.command()
@click.option("--exp_name", type=str, required=True, help="Experiment Name")
@click.option(
    "--exp_dir",
    type=str,
    required=True,
    help="Experiment Directory to store models, checkpoints etc.",
)
@click.option("--src_dom_train_filename", type=str, required=True, help="Source Domain Train File")
@click.option(
    "--src_dom_dev_filename", type=str, required=True, help="Source Domain Development File"
)
@click.option("--src_dom_test_filename", type=str, required=True, help="Source Domain Test File")
@click.option("--trg_dom_train_filename", type=str, required=True, help="Target Domain Train File")
@click.option(
    "--trg_dom_dev_filename", type=str, required=True, help="Target Domain Development File"
)
@click.option("--trg_dom_test_filename", type=str, required=True, help="Target Domain Test File")
@click.option(
    "--max_seq_length",
    type=int,
    required=True,
    help="Sequences will clipped and padded to this length",
)
@click.option("--batch_size", type=int, required=True, help="Batch size for experiments")
@click.option(
    "--enc_emb_size", type=int, required=True, help="Feature extractor(LSTM) embedding size"
)
@click.option(
    "--enc_hidden_size", type=int, required=True, help="Feature extractor(LSTM) hidden size"
)
@click.option(
    "--num_enc_layers", type=int, required=True, help="Feature Extractor(LSTM) number of layers"
)
@click.option(
    "--enc_dropout",
    type=float,
    required=False,
    default=0.0,
    help="Dropout for RNN layers. Applied only when the number of layers is more than 1",
)
@click.option(
    "--is_enc_bidir",
    is_flag=True,
    required=False,
    default=False,
    help="Bidirectional LSTM Encoder. By default true"
)
@click.option(
    "--linear_clf_hidden_dim",
    type=int,
    required=True,
    help="Hidden size for the linear classifier used on top of LSTM"
)
@click.option(
    "--domain_disc_out_dim",
    type=int,
    required=False,
    default=2,
    help="The number of domains for the domain regressor",
)
@click.option(
    "--task_clf_out_dim",
    type=int,
    required=False,
    default=2,
    help="The number of classes of the task. By default 2 for sentiment analysis",
)
@click.option(
    "--dann_alpha", type=float, required=True, help="The alpha parameter of the domain regressor"
)
@click.option(
    "--weight_dom_loss",
    type=float,
    required=True,
    help="Weighs the domain classification loss by this factor",
)
@click.option("--train_proportion", type=float, help="Train on small proportion")
@click.option("--dev_proportion", type=float, help="Validate on small proportion")
@click.option("--test_proportion", type=float, help="Test on small proportion")
@click.option("--seed", type=str, help="Seed for reproducibility")
@click.option("--lr", type=float, help="Learning rate for the entire model")
@click.option("--epochs", type=int, help="Number of epochs to run the training")
@click.option("--gpu", type=int, help="GPU to run the program on")
@click.option("--grad_clip_norm", type=float, help="Gradient Clip Norm value to clip")
@click.option("--wandb_proj_name", type=str, help="Weights and Biases Project Name")
@click.option(
    "--no_adv_train",
    is_flag=True,
    type=bool,
    default=False,
    help="Perform adversarial training or no. If set the adversarial learning is turned off",
)
@click.option(
    "--use_glove_embeddings",
    is_flag=True,
    type=bool,
    default=True,
    help="Load Glove Embeddings and use it for the encoder"
)
@click.option(
    "--glove_name",
    type=str,
    required=False,
    default=None,
    help="The name of the glove embedding"
)
@click.option(
    "--glove_dim",
    type=int,
    required=False,
    default=None,
    help="Glove embedding dimension"
)
@click.option(
    "--max_vocab_size",
    type=int,
    required=False,
    help="Vocab size will be cut down"
)
@click.option(
    "--use_gru_encoder",
    is_flag=True,
    required=False,
    default=False,
    help="If set, uses GRU encoder"
)
def train_dann(
    exp_name,
    exp_dir,
    src_dom_train_filename,
    src_dom_dev_filename,
    src_dom_test_filename,
    trg_dom_train_filename,
    trg_dom_dev_filename,
    trg_dom_test_filename,
    max_seq_length,
    batch_size,
    enc_emb_size,
    enc_hidden_size,
    num_enc_layers,
    enc_dropout,
    is_enc_bidir,
    linear_clf_hidden_dim,
    domain_disc_out_dim,
    task_clf_out_dim,
    dann_alpha,
    weight_dom_loss,
    train_proportion,
    dev_proportion,
    test_proportion,
    seed,
    lr,
    epochs,
    gpu,
    grad_clip_norm,
    wandb_proj_name,
    no_adv_train,
    use_glove_embeddings,
    glove_name,
    glove_dim,
    max_vocab_size,
    use_gru_encoder
):
    hparams = locals()
    seed_everything(seed)

    exp_dir = pathlib.Path(exp_dir)
    if not exp_dir.is_dir():
        exp_dir.mkdir(parents=True)
    else:
        is_delete = Confirm.ask(f"{exp_dir} already exists. Delete it?")
        if is_delete:
            shutil.rmtree(exp_dir)
            console.print(f"[green] deleted {exp_dir}")
        exp_dir.mkdir(parents=True)

    vocab_file = exp_dir.joinpath("vocab.txt")

    hparams["vocab_file"] = str(vocab_file)

    dm = DANNDataModule(hparams=hparams)
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    pad_idx = dm.vocab.get_word2idx()["<pad>"]
    hparams["pad_idx"] = pad_idx
    hparams["idx2token"] = dm.vocab.get_idx2word()
    vocab_size = dm.get_vocab_size()
    hparams["vocab_size"] = vocab_size

    if glove_name and glove_dim:
        glove_embedding = dm.vocab.get_glove_embeddings()
        hparams["glove_embedding"] = glove_embedding

    ######################
    # Model Setup
    ######################
    model = DANN(hparams)

    ###########################################################################
    # SETUP THE LOGGERS and Checkpointers
    ###########################################################################
    logger = WandbLogger(
        name=exp_name,
        save_dir=str(exp_dir),
        project=wandb_proj_name,
    )

    logger.watch(model, log="gradients", log_freq=10)

    callbacks = []

    checkpoints_dir = exp_dir.joinpath("checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        save_top_k=1,
        mode="max",
        monitor="dev/src_acc",
    )
    callbacks.append(checkpoint_callback)

    trainer = Trainer(
        limit_train_batches=train_proportion,
        limit_val_batches=dev_proportion,
        limit_test_batches=test_proportion,
        callbacks=callbacks,
        terminate_on_nan=True,
        gradient_clip_val=grad_clip_norm,
        gpus=str(gpu),
        max_epochs=epochs,
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)

    dm.setup("test")
    test_loader = dm.test_dataloader()
    trainer.test(model, test_loader)

    hparams_file = exp_dir.joinpath("hparams.json")
    if glove_name and glove_dim:
        hparams.pop("glove_embedding")
    with open(hparams_file, "w") as fp:
        json.dump(hparams, fp)


if __name__ == "__main__":
    train_dann()
