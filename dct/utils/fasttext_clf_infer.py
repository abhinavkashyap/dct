import click
import wasabi
import fasttext
from itertools import takewhile
import numpy as np


@click.command()
@click.option(
    "--inp_file",
    type=str,
    help="File containing one sentence per line",
)
@click.option("--model-path", type=str, help="Path where the model is stored")
@click.option("--labels-file", type=str, help="File containing labels one per line")
def fastext_clf_infer(inp_file, model_path, labels_file):
    model = fasttext.load_model(model_path)
    printer = wasabi.Printer()
    sentences = []
    with open(inp_file) as fp:
        for line in fp:
            sentence_ = line.strip()
            sentence_words = sentence_.split()
            sentence_words = list(takewhile(lambda x: x != "<eos>", sentence_words))
            final_sentence_ = " ".join(sentence_words)
            sentences.append(final_sentence_)

    labels = []
    with open(labels_file, "r") as fp:
        for line in fp:
            label = line.strip()
            labels.append(label)

    predictions, _ = model.predict(sentences)
    predictions = list(map(lambda pred: pred[0].replace("__label__", ""), predictions))
    len_predictions = len(predictions)
    num_correct = sum(
        [
            1
            for prediction, correct_label in zip(predictions, labels)
            if prediction == correct_label
        ]
    )
    accuracy = num_correct / len_predictions
    accuracy = np.round(accuracy, 3)
    printer.good(f"accuracy={accuracy * 100}%")


if __name__ == "__main__":
    fastext_clf_infer()
