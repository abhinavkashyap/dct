import pathlib
from typing import List, Tuple
import json
import wasabi
from loguru import logger
import spacy
import numpy as np
from tqdm import tqdm
from dct.data.vocab import Vocabulary


class AmazonDataCleaning:
    def __init__(self, json_filename: pathlib.Path, loguru_logger=None, sample_size: int = None):
        """Download the Jean McAuley amazon review dataset. Every domain is a .gz file
        Use gzip -dk filename.gz to unzip the file

        Parameters
        ----------
        json_filename: pathlib.Pat
            The json file downloaded
        loguru_logger
        sample_size
        """
        self.json_filename = json_filename
        self.loguru_logger = loguru_logger
        self.sample_size = sample_size
        self.msg_printer = wasabi.Printer()

        if self.loguru_logger is None:
            self.loguru_logger = logger

        with self.msg_printer.loading("Loading Spacy"):
            self.nlp = spacy.load(
                "en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"]
            )
            self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))
        self.msg_printer.good(f"Finished Loading Spacy")

    def _read_amazon_review_text(self) -> Tuple[List[str], List[str]]:
        review_texts: List[str] = []
        sentiments: List[str] = []
        no_empty_reviews = 0
        with open(str(self.json_filename), "r") as fp:
            for line in tqdm(fp, desc=f"Reading Review Jsons from {self.json_filename}"):
                review_json = json.loads(line)
                review_text = review_json.get("reviewText")
                review_ratings = review_json.get("overall")

                if review_ratings is None:
                    continue
                if int(review_ratings) == 3:
                    continue
                elif int(review_ratings) < 3:
                    sentiment = "negative"
                else:
                    sentiment = "positive"

                if review_text is not None:
                    review_texts.append(review_text)
                    sentiments.append(sentiment)
                else:
                    no_empty_reviews += 1

        self.msg_printer.good(
            f"Finished reading {len(review_texts)} reviews and {len(sentiments)} sentiments "
            f"from {self.json_filename}"
        )
        self.loguru_logger.info(f"[SKIPPED]: {no_empty_reviews} empty reviews")

        if self.sample_size is not None:
            # randomly choses self.sample_size reviews
            sampled_review_texts = []
            sampled_sentiments = []
            random_ints = np.random.randint(low=0, high=len(review_texts), size=self.sample_size)
            for idx in random_ints:
                sampled_review_texts.append(review_texts[idx])
                sampled_sentiments.append(sentiments[idx])
            return sampled_review_texts, sampled_sentiments
        else:
            return review_texts, sentiments

    def _sent_tokenize(self, review_texts: List[str]) -> List[List[str]]:
        """Tokenizes review text into sentences

        Parameters
        ----------
        review_texts: List[str]
            Possible multiple sentence review texts
        Returns
        -------
        List[List[str]]
            A list of sentences where every sentence is a list of tokens
        """
        sentence_tokens: List[List[str]] = []
        for doc in tqdm(
            self.nlp.pipe(review_texts, batch_size=1000, n_process=40),
            total=len(review_texts),
            desc="Sentence Tokenizing Reviews",
        ):
            sents = doc.sents

            # tokens of every sentence is obtained
            for sent in sents:
                sent_tokens = [str(tok) for tok in sent]
                sentence_tokens.append(sent_tokens)

        self.msg_printer.good(f"Finished Sentence Tokenizing {len(review_texts)} reviews")
        return sentence_tokens

    def write_tokenized_file(self, output_filename: pathlib.Path):
        review_texts, review_sentiments = self._read_amazon_review_text()
        assert len(review_texts) == len(review_sentiments)
        num_sents_written = 0

        out_fp = open(str(output_filename), "w")
        out_sentiments_fp = open(f"{output_filename}.sentiment", "w")
        for sent, sentiment in tqdm(
            zip(review_texts, review_sentiments),
            desc="Writing Reviews and Sentiments",
            total=len(review_texts),
        ):
            text = sent.strip()
            text = text.replace("\n", "")
            if bool(text):
                out_fp.write(text)
                out_fp.write("\n")
                out_sentiments_fp.write(sentiment)
                out_sentiments_fp.write("\n")
                num_sents_written += 1

        self.msg_printer.good(f"Sentences and Sentiments available in {output_filename}")
        self.loguru_logger.info(f"[WROTE] {num_sents_written} sentences to {output_filename}")


def split_reviews(
    filename: str,
    labels_filename: str,
    num_dann_train: int,
    num_dann_dev: int,
    num_dann_test: int,
    num_sentiment_clf_train: int,
    num_sentiment_clf_dev: int,
    num_sentiment_clf_test: int,
    num_transfer_train: int,
    num_transfer_dev: int,
    num_transfer_test: int,
    out_filename_prefix: str,
):
    """Randomly select num_train num_dev and num_test lines
    from the files

    Parameters
    ----------
    filename: str
        Review file where one line contains one instance
    num_dann_train: int
        The number of lines to keep in training
    num_dann_dev: int
        The number of lines in development
    num_dann_test: int
        The number of lines in test
    num_sentiment_clf_train: int
    num_sentiment_clf_dev: int
    num_sentiment_clf_test: int
    num_transfer_train: int
    num_transfer_dev: int
    num_transfer_test: int
    out_filename_prefix: str
        The out fiename prefix

    Returns
    -------
    None
    """
    sentiment_label_mapping = {"positive": 1, "negative": 0}
    count = 0
    reviews = []
    labels = []
    with open(filename, "r") as fp:
        for line in tqdm(fp, desc="Reading Reviews file"):
            line_ = line.strip()
            reviews.append(line_)
            count += 1

    with open(labels_filename, "r") as fp:
        for line in tqdm(fp, desc="Reading Labels file"):
            line_ = line.strip()
            labels.append(line_)

    assert count > (
        num_dann_train
        + num_dann_dev
        + num_dann_test
        + num_sentiment_clf_train
        + num_sentiment_clf_dev
        + num_sentiment_clf_test
        + num_transfer_train
        + num_transfer_dev
        + num_transfer_test
    ), f"Not enough lines to sample from file"

    numbers_range = range(count)
    num_samples = (
        num_dann_train
        + num_dann_dev
        + num_dann_test
        + num_sentiment_clf_train
        + num_sentiment_clf_dev
        + num_sentiment_clf_test
        + num_transfer_train
        + num_transfer_dev
        + num_transfer_test
    )

    np.random.seed(1729)
    samples = np.random.choice(numbers_range, size=num_samples, replace=False)

    # Total number of train samples
    num_train_samples = num_dann_train + num_sentiment_clf_train + num_transfer_train
    num_dev_samples = num_dann_dev + num_sentiment_clf_dev + num_transfer_dev
    num_test_samples = num_dann_test + num_sentiment_clf_test + num_transfer_test

    #  Get train, dev and test indices for the the three kinds of tasks
    all_train_indices = samples[:num_train_samples]
    all_dev_indices = samples[num_train_samples : (num_train_samples + num_dev_samples)]
    all_test_indices = samples[(num_train_samples + num_dev_samples) :]

    # Get train indices for every kind of task
    dann_train_samples = all_train_indices[:num_dann_train]
    sentiment_clf_train_samples = all_train_indices[
        num_dann_train : (num_dann_train + num_sentiment_clf_train)
    ]
    transfer_train_samples = all_train_indices[(num_dann_train + num_sentiment_clf_train) :]

    # Get dev indices for every kind of task
    dann_dev_samples = all_dev_indices[:num_dann_dev]
    sentiment_clf_dev_samples = all_dev_indices[
        num_dann_dev : (num_dann_dev + num_sentiment_clf_dev)
    ]
    transfer_dev_samples = all_dev_indices[(num_dann_dev + num_sentiment_clf_dev) :]

    # Get test indices for every kind of task

    dann_test_samples = all_test_indices[:num_dann_test]
    sentiment_clf_test_samples = all_test_indices[
        num_dann_test : (num_dann_test + num_sentiment_clf_test)
    ]
    transfer_test_samples = all_test_indices[(num_dann_test + num_sentiment_clf_test) :]

    # Define dann filenames
    dann_train_filename = f"{out_filename_prefix}.dann.train"
    dann_dev_filename = f"{out_filename_prefix}.dann.dev"
    dann_test_filename = f"{out_filename_prefix}.dann.test"

    dann_train_fp = open(dann_train_filename, "w")
    dann_dev_fp = open(dann_dev_filename, "w")
    dann_test_fp = open(dann_test_filename, "w")

    for index in dann_train_samples:
        review = reviews[index]
        label = labels[index]
        label = sentiment_label_mapping[label]
        # \042 is double quotes
        line_ = f"\042{review}\042###{label}"
        dann_train_fp.write(line_)
        dann_train_fp.write("\n")

    for index in dann_dev_samples:
        review = reviews[index]
        label = labels[index]
        label = sentiment_label_mapping[label]
        # \042 is double quotes
        line_ = f"\042{review}\042###{label}"
        dann_dev_fp.write(line_)
        dann_dev_fp.write("\n")

    for index in dann_test_samples:
        review = reviews[index]
        label = labels[index]
        label = sentiment_label_mapping[label]
        # \042 is double quotes
        line_ = f"\042{review}\042###{label}"
        dann_test_fp.write(line_)
        dann_test_fp.write("\n")

    dann_train_fp.close()
    dann_dev_fp.close()
    dann_test_fp.close()

    sentiment_clf_train_filename = f"{out_filename_prefix}.sentimentclf.train"
    sentiment_clf_dev_filename = f"{out_filename_prefix}.sentimentclf.dev"
    sentiment_clf_test_filename = f"{out_filename_prefix}.sentimentclf.test"

    sentiment_clf_train_fp = open(sentiment_clf_train_filename, "w")
    sentiment_clf_dev_fp = open(sentiment_clf_dev_filename, "w")
    sentiment_clf_test_fp = open(sentiment_clf_test_filename, "w")

    for index in sentiment_clf_train_samples:
        review = reviews[index]
        label = labels[index]
        label = sentiment_label_mapping[label]
        # \042 is double quotes
        line_ = f"\042{review}\042###{label}"
        sentiment_clf_train_fp.write(line_)
        sentiment_clf_train_fp.write("\n")

    for index in sentiment_clf_dev_samples:
        review = reviews[index]
        label = labels[index]
        label = sentiment_label_mapping[label]
        # \042 is double quotes
        line_ = f"\042{review}\042###{label}"
        sentiment_clf_dev_fp.write(line_)
        sentiment_clf_dev_fp.write("\n")

    for index in sentiment_clf_test_samples:
        review = reviews[index]
        label = labels[index]
        label = sentiment_label_mapping[label]
        # \042 is double quotes
        line_ = f"\042{review}\042###{label}"
        sentiment_clf_test_fp.write(line_)
        sentiment_clf_test_fp.write("\n")

    sentiment_clf_train_fp.close()
    sentiment_clf_dev_fp.close()
    sentiment_clf_test_fp.close()

    transfer_train_filename = f"{out_filename_prefix}.transfer.train"
    transfer_dev_filename = f"{out_filename_prefix}.transfer.dev"
    transfer_test_filename = f"{out_filename_prefix}.transfer.test"

    transfer_train_fp = open(transfer_train_filename, "w")
    transfer_dev_fp = open(transfer_dev_filename, "w")
    transfer_test_fp = open(transfer_test_filename, "w")
    transfer_train_sentiment_fp = open(f"{transfer_train_filename}.sentiment.txt", "w")
    transfer_dev_sentiment_fp = open(f"{transfer_dev_filename}.sentiment.txt", "w")
    transfer_test_sentiment_fp = open(f"{transfer_test_filename}.sentiment.txt", "w")

    for index in transfer_train_samples:
        review = reviews[index]
        transfer_train_fp.write(review)
        transfer_train_fp.write("\n")
        label = labels[index]
        label = sentiment_label_mapping[label]
        label = str(label)
        transfer_train_sentiment_fp.write(label)
        transfer_train_sentiment_fp.write("\n")

    for index in transfer_dev_samples:
        review = reviews[index]
        transfer_dev_fp.write(review)
        transfer_dev_fp.write("\n")
        label = labels[index]
        label = sentiment_label_mapping[label]
        label = str(label)
        transfer_dev_sentiment_fp.write(label)
        transfer_dev_sentiment_fp.write("\n")

    for index in transfer_test_samples:
        review = reviews[index]
        transfer_test_fp.write(review)
        transfer_test_fp.write("\n")
        label = labels[index]
        label = sentiment_label_mapping[label]
        label = str(label)
        transfer_test_sentiment_fp.write(label)
        transfer_test_sentiment_fp.write("\n")


def form_vocab(
    src_train_filename,
    trg_train_filename,
    src_dev_filename,
    trg_dev_filename,
    max_vocab_size,
    vocab_filename,
    max_seq_length
):
    with open(src_train_filename) as fp:
        src_train_tokens = []
        for line in tqdm(fp, desc="Reading src-train tokens"):
            line_ = line.strip()
            line_ = line_.replace('"', "")
            line_ = line_.lower()
            tokens = line_.split()
            tokens = tokens[:max_seq_length]
            src_train_tokens.append(tokens)

    with open(src_dev_filename) as fp:
        src_dev_tokens = []
        for line in tqdm(fp, desc="Reading src-dev tokens"):
            line_ = line.strip()
            line_ = line_.replace('"', "")
            line_ = line_.lower()
            tokens = line_.split()
            tokens = tokens[:max_seq_length]
            src_dev_tokens.append(tokens)

    with open(trg_train_filename) as fp:
        trg_train_tokens = []
        for line in tqdm(fp, desc="Reading trg-train tokens"):
            line_ = line.strip()
            line_ = line_.replace('"', "")
            line_ = line_.lower()
            tokens = line_.split()
            tokens = tokens[:max_seq_length]
            trg_train_tokens.append(tokens)

    with open(trg_dev_filename) as fp:
        trg_dev_tokens = []
        for line in tqdm(fp, desc="Reading trg-dev tokens"):
            line_ = line.strip()
            line_ = line_.replace('"', "")
            line_ = line_.lower()
            tokens = line_.split()
            tokens = tokens[:max_seq_length]
            trg_dev_tokens.append(tokens)

    vocab = Vocabulary(
        instances=src_train_tokens + src_dev_tokens + trg_train_tokens + trg_dev_tokens,
        max_vocab_size=max_vocab_size,
        add_special_tokens=True,
    )
    vocab.save_vocab(vocab_filename)
    vocab.print_statistics()


if __name__ == "__main__":
    json_filename = pathlib.Path("/abhinav/dct/data/mcauley_reviews/Electronics.json")
    output_filename = pathlib.Path("/abhinav/dct/data/mcauley_reviews/dvd.txt")
    # amazon_cleaner = AmazonDataCleaning(json_filename=json_filename, sample_size=1000000)
    # amazon_cleaner.write_tokenized_file(output_filename=output_filename)
    # split_reviews(
    #     str(output_filename),
    #     labels_filename="/abhinav/dct/data/mcauley_reviews/dvd.txt.sentiment",
    #     num_dann_train=2000,
    #     num_dann_dev=500,
    #     num_dann_test=500,
    #     num_sentiment_clf_train=2000,
    #     num_sentiment_clf_dev=500,
    #     num_sentiment_clf_test=500,
    #     num_transfer_train=100000,
    #     num_transfer_dev=1000,
    #     num_transfer_test=1000,
    #     out_filename_prefix="/abhinav/dct/data/mcauley_reviews/dvd",
    # )

    form_vocab(
        src_train_filename="/abhinav/dct/data/mcauley_reviews/dvd.dannplustransfer.train",
        src_dev_filename="/abhinav/dct/data/mcauley_reviews/dvd.dannplustransfer.dev",
        trg_train_filename="/abhinav/dct/data/mcauley_reviews/electronics.dannplustransfer.train",
        trg_dev_filename="/abhinav/dct/data/mcauley_reviews/electronics.dannplustransfer.dev",
        max_vocab_size=100000,
        vocab_filename="/abhinav/dct/data/mcauley_reviews/vocab.txt",
        max_seq_length=20,
    )
