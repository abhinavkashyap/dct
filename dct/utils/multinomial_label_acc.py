import pathlib
from dct.utils.generate_constraints import GenerateLabels
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
from itertools import takewhile
from typing import Union


class MultinomialLabelAccuracy:
    def __init__(
        self,
        pred_filename: Union[pathlib.Path, str],
        true_filename: Union[pathlib.Path, str],
        avg_length: int,
        src_dom_saliency_file: Union[pathlib.Path, str],
        trg_dom_saliency_file: Union[pathlib.Path, str],
    ):
        """

        Parameters
        ----------
        pred_filename: pathlib.Path
            File with one string per line
        true_filename: pathlib.Path
            File with one string per line
        avg_length: int
            This length will be used for binarizing the length
            labels oft he lines
        """
        self.pred_filename = pred_filename
        self.true_filename = true_filename
        self.avg_legnth = avg_length
        self.src_dom_saliency_file = src_dom_saliency_file
        self.trg_dom_saliency_file = trg_dom_saliency_file
        self.generate_labels = GenerateLabels(
            avg_length=avg_length,
            src_dom_saliency_file=src_dom_saliency_file,
            trg_dom_saliency_file=trg_dom_saliency_file,
        )

        self.pred_lines = self.read_instances(filename=self.pred_filename)
        self.true_lines = self.read_instances(filename=self.true_filename)

        true_labels = self.generate_labels.from_lines(self.pred_lines)
        pred_labels = self.generate_labels.from_lines(self.true_lines)

        self.label_names = list(true_labels[0].keys())

        all_true_labels = []

        for line_label in true_labels:
            line_label_ = list(line_label.values())
            all_true_labels.append(line_label_)

        all_pred_labels = []
        for line_label in pred_labels:
            line_label_ = list(line_label.values())
            all_pred_labels.append(line_label_)

        self.true_labels = all_true_labels
        self.pred_labels = all_pred_labels
        self.true_labels = np.array(self.true_labels)
        self.pred_labels = np.array(self.pred_labels)

    @staticmethod
    def read_instances(filename):
        lines = []
        with open(filename, "r") as fp:
            for line in fp:
                line_ = line.strip().split()
                line_ = takewhile(lambda word: word != "<eos>", line_)
                line_ = " ".join(line_)
                lines.append(line_)

        return lines

    def get_accuracy(self):
        # get accuracy for every column in the true labels and pred_labels
        results = {}
        ncols = self.true_labels.shape[1]
        for col in range(ncols):
            true_labels_ = self.true_labels[:, col]
            pred_labels_ = self.pred_labels[:, col]
            label_name = self.label_names[col]
            p, r, f, _ = precision_recall_fscore_support(
                true_labels_, pred_labels_, average="macro"
            )
            acc = accuracy_score(true_labels_, pred_labels_)
            print(f"label_name {label_name}, pr: {p}, recall: {r} f_measure:{f}, acc: {acc}")
            results[label_name] = {"precision": p, "recall": r, "f-score": f, "acc": acc}
        return results


# if __name__ == "__main__":
#     src_dom_saliency_file = "/home/rkashyap/abhi/synarae/data/compositional_data/src_dom.attributes.txt"
#     trg_dom_saliency_file = "/home/rkashyap/abhi/synarae/data/compositional_data/trg_dom.attributes.txt"
#     prediction_filename = "/home/rkashyap/abhi/arae/yelp/composition_output/26_output_decoder_1_tran.txt"
#     truelabels_filename = "/home/rkashyap/abhi/arae/yelp/composition_output/26_output_decoder_1_from.txt"
#
#     label_accuracies = MultinomialLabelAccuracy(
#         pred_filename=prediction_filename,
#         true_filename=truelabels_filename,
#         avg_length=10,
#         src_dom_saliency_file=src_dom_saliency_file,
#         trg_dom_saliency_file=trg_dom_saliency_file
#
#     )
#     results = label_accuracies.get_accuracy()
#     print(results)
