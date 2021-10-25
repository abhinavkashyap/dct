"""
Classifies text based on the style (usually binary classification)
This is necessary to calculate the transfer accuracy of the models
"""
import click
import pathlib
import numpy as np
import fasttext
import shutil
import wasabi
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


@click.command()
@click.option(
    "--train-file",
    type=str,
    help="File containing one sentence per line for class 1",
)
@click.option(
    "--dev-file",
    type=str,
    help="File containing one sentence per line for class 2",
)
@click.option(
    "--test-file",
    type=str,
    help="File containing one sentence per line for testing",
)
@click.option("--model-dir", type=str, help="Folder to store the trained models")
@click.option(
    "--model-name", default="supervised_clf.model", help="The name of the model file"
)
def fasttext_clf(
    train_file,
    dev_file,
    test_file,
    model_dir,
    model_name,
):
    np.random.seed(1729)
    train_file = pathlib.Path(train_file)
    dev_file = pathlib.Path(dev_file)
    test_file = pathlib.Path(test_file)
    model_dir = pathlib.Path(model_dir)
    msg_printer = wasabi.Printer()

    if not model_dir.is_dir():
        model_dir.mkdir(parents=True)

    tmp_dir = pathlib.Path("./fasttext_tmp")
    if not tmp_dir.is_dir():
        tmp_dir.mkdir(parents=True)

    # we will sample equal number of lines from both files
    train_lines = []
    test_lines = []

    with open(str(train_file)) as fp:
        for line in fp:
            line_, label_ = line.split("###")
            line_ = line_.strip()
            # remove the first and the last `""`
            line_ = line_[1:-1]
            label_ = label_.strip()
            instance = " ".join([f"__label__{label_}", line_])
            train_lines.append(instance)

    test_true = []
    with open(str(test_file)) as fp:
        for line in fp:
            line_, label_ = line.split("###")
            line_ = line_.strip()
            # remove the first and the last `""`
            line_ = line_[1:-1]
            label_ = label_.strip()
            test_lines.append(line_)
            test_true.append(label_)

    train_file = tmp_dir.joinpath(f"style_clf_train.txt")
    with open(str(train_file), "w") as fp:
        for line in train_lines:
            fp.write(line)
            fp.write("\n")

    model = fasttext.train_supervised(str(train_file), seed=1729, epoch=100, lr=0.1)
    model.save_model(str(model_dir.joinpath(f"{model_name}")))
    _, train_prec, train_rec = model.test(str(train_file))
    print(f"Training Metrics")
    print(f"=" * 80)
    print(f"Precision {train_prec}")
    print(f"Recall {train_rec}")

    test_pred = []
    test_predictions = model.predict(test_lines)[0]
    test_predictions = list(map(lambda pred: pred[0].replace("__label__", ""), test_predictions))
    test_pred.extend(test_predictions)

    acc = accuracy_score(test_true, test_pred)
    p, r, f, _ = precision_recall_fscore_support(test_true, test_pred, average="macro")

    print(f"Test Metrics")
    print(f"=" * 80)
    print(f"prec: {p}, recall: {r}, fscore: {f} accuracy: {acc}")

    # remove the tmp directory created
    shutil.rmtree(str(tmp_dir))


if __name__ == "__main__":
    fasttext_clf()
