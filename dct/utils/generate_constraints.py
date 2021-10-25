import pathlib
from typing import Dict, Any, Optional, List
import click
from collections import OrderedDict
import spacy
from tqdm import tqdm
from dct.utils.data_utils import get_average_heights
from dct.utils.data_utils import ngram_tokenize
from collections import Counter
import pandas as pd
import json


class GenerateLabels:
    def __init__(
        self,
        avg_length: int,
        src_dom_saliency_file: Optional[pathlib.Path],
        trg_dom_saliency_file: Optional[pathlib.Path],
        filename: Optional[pathlib.Path] = None,
        out_label_filename: Optional[pathlib.Path] = None,
        ngram_range: int = 5,
    ):
        """
        This is used to generate corresponding labels for text
        Parameters
        ----------
        filename: pathlib.Path
            The path to the file
        avg_length: int
            The average length of the sentence below which
            it is consider as short sentences and above which it is considered
            as longer sentences
        src_dom_saliency_file: pathlib.Path
            The file containing saliency words from the source domain
        trg_dom_saliency_file: pathlib.Path
            The file containing saliency words from the target domain
        out_label_filename: pathlib.Path
            The output label filename where one dictionary
            containing the labels for a sentence is stored
        ngram_range: int
            ngram range used to calcualte the saliency
        """
        self.filename = filename
        self.avg_length = avg_length
        self.adjectives_limit = 5
        self.syntax_tree_height = 10
        self.prop_noun_limit = 3
        self.out_label_filename = out_label_filename
        self.src_dom_saliency_file = src_dom_saliency_file
        self.trg_dom_saliency_file = trg_dom_saliency_file
        self.src_dom_salient_words = set(
            self._read_instances(self.src_dom_saliency_file)
        )
        self.trg_dom_salient_words = set(
            self._read_instances(self.trg_dom_saliency_file)
        )
        self.ngram_range = ngram_range

        if self.filename is not None:
            self.lines = self._read_instances(self.filename)

        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

    @staticmethod
    def _read_instances(filename):
        lines = []
        with open(str(filename)) as fp:
            for line in fp:
                line_ = line.strip()
                lines.append(line_)

        return lines

    def gen_labels(self):
        return self._gen_labels(self.lines)

    def _gen_labels(self, lines):
        length_labels = self._gen_length_category_labels(lines)
        spacy_related_labels = self._get_spacy_related_labels(lines)
        num_dom_specific_attrs_labels = self._get_number_dom_specific_entities_labels(
            lines
        )
        labels = []

        for length_label_dict, spacy_labels_dict, dom_specific_attr_label_dict in zip(
            length_labels, spacy_related_labels, num_dom_specific_attrs_labels
        ):
            labels.append(
                {
                    **length_label_dict,
                    **spacy_labels_dict,
                    **dom_specific_attr_label_dict,
                }
            )

        return labels

    def _gen_length_category_labels(self, lines: str) -> List[Dict[str, Any]]:

        labels = []
        for line in tqdm(
            lines, total=len(lines), desc="Gen Length Labels", leave=False
        ):
            words = line.strip().split()
            len_words = len(words)

            # is 0 when the sentence is shorter than the average length
            # else it is 1
            length_category = "long" if len_words >= self.avg_length else "short"

            labels.append(OrderedDict({"length": str(length_category)}))
        return labels

    def _get_spacy_related_labels(self, lines) -> List[Dict[str, Any]]:
        # is 1 when the sentence contains a personal pronoun
        # otherwise it is 0
        labels = []
        for doc in tqdm(
            self.nlp.pipe(
                lines,
                disable=["ner", "lemmatizer", "tok2vec", "attribute_ruler"],
            ),
            desc="Generating Syntax Labels",
            total=len(lines),
            leave=False,
        ):
            is_personal_pronoun = "not_personal"
            pos_tags = []
            num_tokens = 0
            if len(doc) == 0:
                doc_avg_height = 1
            else:
                doc_avg_height = int(get_average_heights(doc))
            for token in doc:
                pos = token.pos_
                pos_tags.append(pos)
                num_tokens += 1
                pronoun = token.morph.pron_type_
                if pronoun != "":
                    pronoun_type = pronoun.split("_")[-1]
                    if pronoun_type == "prs" and token.text.lower() in ["i", "my"]:
                        is_personal_pronoun = "personal"
                        break

            adj_tags = list(filter(lambda tag: tag == "ADJ", pos_tags))
            num_adjective = len(adj_tags)

            if num_adjective < self.adjectives_limit:
                descriptive = f"num_adj_{num_adjective}"
            else:
                descriptive = f"num_adj_long"

            if doc_avg_height < self.syntax_tree_height:
                doc_avg_height = f"height_{doc_avg_height}"
            else:
                doc_avg_height = f"height_long"

            prop_noun_tags = list(filter(lambda tag: tag == "PROPN", pos_tags))
            num_prop_noun_tags = len(prop_noun_tags)

            if num_prop_noun_tags < self.prop_noun_limit:
                prop_noun_tag_label = f"prop_noun_{num_prop_noun_tags}"
            else:
                prop_noun_tag_label = f"prop_noun_tag_long"

            labels.append(
                OrderedDict(
                    {
                        "personal": is_personal_pronoun,
                        "descriptive": descriptive,
                        "tree_height": doc_avg_height,
                        "prop_noun": prop_noun_tag_label,
                    }
                )
            )
        return labels

    def _get_number_dom_specific_entities_labels(self, lines) -> List[Dict[str, Any]]:
        labels = []
        for line in tqdm(
            lines, desc="Number of domain specific attributes", total=len(lines)
        ):
            ngrams = ngram_tokenize(text=line, ngram_range=int(self.ngram_range))
            attributes = []
            for gram in ngrams:
                if (
                    gram in self.src_dom_salient_words
                    or gram in self.trg_dom_salient_words
                ):
                    attributes.append(gram.strip())

            num_attributes = len(attributes)
            label_num_attributes = (
                f"dom_attr_{num_attributes}" if num_attributes < 5 else f"dom_attr_long"
            )
            labels.append(OrderedDict({"num_dom_attributes": label_num_attributes}))

        return labels

    def write_labels(self):
        out_label_filename = str(self.out_label_filename)
        with open(out_label_filename, "w") as fp:
            labels = self.gen_labels()
            for label in labels:
                fp.write(json.dumps(label))
                fp.write("\n")

        label_distribution = self.get_distributions(labels)
        with open(f"{out_label_filename}.distributions", "w") as fp:
            json.dump(label_distribution, fp)

    def from_lines(self, lines: List[str]):
        return self._gen_labels(lines)

    @staticmethod
    def get_distributions(labels: List[Dict[str, Any]]):
        label_names = list(labels[0].keys())
        df = pd.DataFrame(labels)

        # mapping from a label name to the distribution
        distribution = {}
        for label_name in label_names:
            label_values = df[label_name].tolist()
            label_distribution = Counter(label_values)
            distribution[label_name] = label_distribution

        return distribution


@click.command()
@click.option(
    "--filename", type=str, help="Yelp filename containing one sentence per line"
)
@click.option(
    "--out-label-filename", type=str, help="The labels will be written to this file"
)
@click.option(
    "--avg-length",
    type=int,
    help="The averge length of the sentences to consider while "
    "classifying the sentence as short or long",
)
@click.option(
    "--src-dom-saliency-file", type=str, help="src domain salient attributes file"
)
@click.option(
    "--trg-dom-saliency-file", type=str, help="trg domain salient attributes file"
)
def gen_constraints(
    filename,
    out_label_filename,
    avg_length,
    src_dom_saliency_file,
    trg_dom_saliency_file,
):
    filename = pathlib.Path(filename)
    out_label_filename = pathlib.Path(out_label_filename)
    generator = GenerateLabels(
        filename=filename,
        out_label_filename=out_label_filename,
        avg_length=avg_length,
        src_dom_saliency_file=src_dom_saliency_file,
        trg_dom_saliency_file=trg_dom_saliency_file,
    )
    generator.write_labels()


if __name__ == "__main__":
    gen_constraints()
