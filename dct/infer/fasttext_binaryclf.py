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
    "--class1-train-file",
    type=str,
    help="File containing one sentence per line for class 1",
)
@click.option(
    "--class2-train-file",
    type=str,
    help="File containing one sentence per line for class 2",
)
@click.option(
    "--class1-test-file",
    type=str,
    help="File containing one sentence per line for testing",
)
@click.option(
    "--class2-test-file",
    type=str,
    help="File containing one sentence per line for testing",
)
@click.option("--model-dir", type=str, help="Folder to store the trained models")
@click.option(
    "--model-name", default="supervised_clf.model", help="The name of the model file"
)
def fasttext_clf(
    class1_train_file,
    class2_train_file,
    class1_test_file,
    class2_test_file,
    model_dir,
    model_name,
):
    np.random.seed(1729)
    class1_train_file = pathlib.Path(class1_train_file)
    class2_train_file = pathlib.Path(class2_train_file)
    class1_test_file = pathlib.Path(class1_test_file)
    class2_test_file = pathlib.Path(class2_test_file)
    model_dir = pathlib.Path(model_dir)
    msg_printer = wasabi.Printer()

    if not model_dir.is_dir():
        model_dir.mkdir(parents=True)

    tmp_dir = pathlib.Path("./fasttext_tmp")
    if not tmp_dir.is_dir():
        tmp_dir.mkdir(parents=True)

    # we will sample equal number of lines from both files
    class1_train_lines = []
    class2_train_lines = []
    class1_test_lines = []
    class2_test_lines = []

    with open(str(class1_train_file)) as fp:
        for line in fp:
            line_ = line.strip()
            class1_train_lines.append(line_)

    with open(str(class2_train_file)) as fp:
        for line in fp:
            line_ = line.strip()
            class2_train_lines.append(line_)

    with open(str(class1_test_file)) as fp:
        for line in fp:
            line_ = line.strip()
            class1_test_lines.append(line_)

    with open(str(class2_test_file)) as fp:
        for line in fp:
            line_ = line.strip()
            class2_test_lines.append(line_)

    num_lines_class1 = len(class1_train_lines)
    num_lines_class2 = len(class2_train_lines)

    min_num_lines = min(num_lines_class1, num_lines_class2)

    msg_printer.info(f"Training on {2*min_num_lines} from the two files")

    class1_train_lines = np.random.choice(
        class1_train_lines,
        size=min_num_lines,
        replace=False,
    )
    class2_train_lines = np.random.choice(
        class2_train_lines, size=min_num_lines, replace=False
    )

    # map every string to fast text string
    class1_train_lines = map(
        lambda text: " ".join(["__label__1", text]), class1_train_lines
    )
    class2_train_lines = map(
        lambda text: " ".join(["__label__2", text]), class2_train_lines
    )

    class1_train_lines = list(class1_train_lines)
    class2_train_lines = list(class2_train_lines)

    train_lines = class1_train_lines + class2_train_lines
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

    test_true = []
    test_pred = []
    class1_test_predictions = model.predict(class1_test_lines)[0]
    class1_test_predictions = list(map(lambda pred: pred[0], class1_test_predictions))
    test_true.extend(["__label__1"] * len(class1_test_predictions))
    test_pred.extend(class1_test_predictions)

    class2_test_predictions = model.predict(class2_test_lines)[0]
    class2_test_predictions = list(map(lambda pred: pred[0], class2_test_predictions))
    test_true.extend(["__label__2"] * len(class2_test_predictions))
    test_pred.extend(class2_test_predictions)

    acc = accuracy_score(test_true, test_pred)
    p, r, f, _ = precision_recall_fscore_support(test_true, test_pred)

    print(f"Test Metrics")
    print(f"=" * 80)
    print(f"prec: {p}, recall: {r}, fscore: {f} accuracy: {acc}")

    # remove the tmp directory created
    shutil.rmtree(str(tmp_dir))


if __name__ == "__main__":
    fasttext_clf()
