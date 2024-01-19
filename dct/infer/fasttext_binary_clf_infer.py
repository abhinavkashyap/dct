import click
import fasttext
from itertools import takewhile
import wasabi
import numpy as np


@click.command()
@click.option(
    "--ft-model-path", type=str, help="The file path where fast text is stored"
)
@click.option(
    "--sentences-file",
    type=str,
    help="File path where the sentences (one per line) are stored. The transfer accuracy will be reported for this file",
)
@click.option(
    "--correct-label",
    type=str,
    help="What is the correct label for every text in this file?",
)
def fasttext_transfer_acc(ft_model_path, sentences_file, correct_label):
    model = fasttext.load_model(ft_model_path)
    printer = wasabi.Printer()
    sentences = []
    with open(sentences_file) as fp:
        for line in fp:
            sentence_ = line.strip()
            sentence_words = sentence_.split()
            sentence_words = list(takewhile(lambda x: x != "<eos>", sentence_words))
            final_sentence_ = " ".join(sentence_words)
            sentences.append(final_sentence_)

    predictions, _ = model.predict(sentences)
    predictions = list(map(lambda pred: pred[0], predictions))
    len_predictions = len(predictions)
    num_correct = sum([1 for prediction in predictions if prediction == correct_label])
    accuracy = num_correct / len_predictions
    accuracy = np.round(accuracy, 3)
    printer.good(f"accuracy={accuracy * 100}%")


if __name__ == "__main__":
    fasttext_transfer_acc()
