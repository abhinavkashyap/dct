import click
import pathlib
from dct.infer.cola_roberta_infer import ColaRobertaInfer
import numpy as np
import wasabi


@click.command()
@click.option("--file", type=str, help="File where one sentence is stored per line.")
@click.option(
    "--robertaft-checkpoint-dir",
    type=str,
    help="Checkpoint directory where roberta finetuned model is stored",
)
@click.option(
    "--robertaft-params-file",
    type=str,
    help="Hyper-params file used to finetune the roberta model",
)
def yelp_cola_acceptability(file, robertaft_checkpoint_dir, robertaft_params_file):
    printer = wasabi.Printer()
    filename = pathlib.Path(file)
    robertaft_checkpoint_dir = pathlib.Path(robertaft_checkpoint_dir)
    robertaft_params_file = pathlib.Path(robertaft_params_file)

    infer = ColaRobertaInfer(
        checkpoints_dir=robertaft_checkpoint_dir, hparams_file=robertaft_params_file
    )

    with open(filename) as fp:
        lines = []
        for line in fp:
            lines.append(line.strip())

    predictions = infer.predict(lines).tolist()

    percentage = np.mean(predictions)
    printer.good(f"Acc %: {percentage}")

    return predictions, percentage


if __name__ == "__main__":
    yelp_cola_acceptability()
