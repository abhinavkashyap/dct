import click
from dct.infer.pl_arae_infer import ARAEInfer
from dct.infer.pl_arae_contra_infer import ARAEContraInfer
from dct.console import console
import pathlib


@click.command()
@click.option("--checkpoints-dir", type=str, required=True)
@click.option("--hparams-file", type=str, required=True)
@click.option("--from-file", type=str, required=True)
@click.option("--out-file", type=str, required=True)
@click.option("--greedy", is_flag=True)
@click.option("--nucleus-sampling", is_flag=True)
@click.option("--top-p", type=float, default=0.9)
@click.option(
    "--is-contrastive", type=bool, is_flag=True, help="Set this to use contrastive learning models"
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="temperature for scaling the logits during decoding",
)
def write_gen2file(
    checkpoints_dir,
    hparams_file,
    from_file,
    out_file,
    greedy,
    nucleus_sampling,
    top_p,
    is_contrastive,
    temperature,
):

    checkpoints_dir = pathlib.Path(checkpoints_dir)
    hparams_file = pathlib.Path(hparams_file)

    if not checkpoints_dir.is_dir():
        raise ValueError(f"{checkpoints_dir} does not exist")

    if not hparams_file.is_file():
        raise ValueError(f"{hparams_file} does not exist")

    if is_contrastive:
        infer = ARAEContraInfer(checkpoints_dir=checkpoints_dir, hparams_file=hparams_file)
    else:
        print(f"not contrastive infer")
        infer = ARAEInfer(checkpoints_dir=checkpoints_dir, hparams_file=hparams_file)

    with open(from_file, "r") as fp:
        lines = [line.strip() for line in fp]
        transferred_sentences = infer.predict(
            lines=lines,
            nucleus_sampling=nucleus_sampling,
            gen_greedy=greedy,
            top_p=top_p,
            temperature=temperature,
        )

    with open(out_file, "w") as fp:
        for gen_sent in transferred_sentences:
            fp.write(gen_sent)
            fp.write("\n")

    console.print(f"Wrote {out_file}", style="green")


if __name__ == "__main__":
    write_gen2file()
