from bs4 import BeautifulSoup
from typing import List
import bs4
from tqdm import tqdm
import pathlib
from sklearn.model_selection import train_test_split
from dct.console import console
import click
import unicodedata


class BlitzerReviewsCleaning:
    def __init__(self, review_filename: str):
        """Cleans reviews that are downloaded from
        https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
        The dataset is used in Domain Adversarial Neural Networks.
        Download the unprocessed.tar.gz and uncompress it

        Parameters
        ----------
        review_filename: str
            Filename where reviews are stored
        """
        self.review_filename = review_filename

        # make the soup
        with open(self.review_filename, "r", encoding="ISO-8859-1") as fp:
            doc = fp.read()

        self.soup = BeautifulSoup(doc, features="html.parser")

    def _clean_text(self, text: str) -> str:
        """Clean review_text

        Parameters
        ----------
        text: str
            String to be cleaned

        Returns
        -------
        str
            Cleaned string

        """
        text = text.strip()
        text = text.replace("\n", "")
        text = text.lower()
        return text

    def get_review_tags(self) -> List[bs4.element.Tag]:
        """Get <review> elements from the soup

        Returns
        -------
        List[bs4.element.Tag]
            Object representing a tag

        """
        return self.soup.find_all("review")

    def _get_review_texts(self, review_tags: List[bs4.element.Tag]) -> List[str]:
        review_texts = []
        for review_tag in review_tags:
            review_texts.append(review_tag.review_text.string)
        return review_texts

    def get_reviews(self) -> List[str]:
        review_tags = self.get_review_tags()
        review_texts = self._get_review_texts(review_tags)
        cleaned_texts = map(self._clean_text, tqdm(review_texts, desc="Cleaning Reviews"))
        cleaned_texts = map(self.unicodeToString, tqdm(cleaned_texts, desc="Normalizing encoding"))
        cleaned_texts = list(cleaned_texts)
        return cleaned_texts

    # Convert unicode corpus to strings
    @staticmethod
    def unicodeToString(txt: List[str]):
        str_doc = str(unicodedata.normalize("NFKD", txt).encode("ascii", "ignore"))
        return str_doc


@click.command()
@click.option(
    "--dataset_folder_path", type=str, required=True, help="The folder containing the review files"
)
@click.option(
    "--output_folder",
    type=str,
    required=True,
    help="The .train .dev and .test files will be placed in this folder",
)
@click.option(
    "--file_prefix",
    type=str,
    required=True,
    help="The file prefix for the train dev and test files",
)
def form_blitzer_datasets(dataset_folder_path: str, output_folder: str, file_prefix: str):
    """Form the train dev and test split for the Blitzer
    amazon review dataset and write the corresponding
    file_prefix.train file_prefix.dev and file_prefix.test files
    into the output_folder

    Examples
    --------
    form_blitzer_datasets("/path/to/folder/dvd", "/path/to/output_folder")
    Writes the file_prefix.train file_prefix.dev and file_prefix.test
    to the output_folder

    Parameters
    ----------
    dataset_folder_path: str
        The folder containing the review files
    output_folder: str
        The .train .dev and .test files will be placed in this folder
    file_prefix: str
        The file prefix for the train dev and test files
    Returns
    -------
    None

    """
    dataset_folder_path = pathlib.Path(dataset_folder_path)
    positive_reviews_filename = str(dataset_folder_path.joinpath("positive.review"))
    negative_reviews_filename = str(dataset_folder_path.joinpath("negative.review"))
    positive_reviews_cleaner = BlitzerReviewsCleaning(review_filename=positive_reviews_filename)
    negative_reviews_cleaner = BlitzerReviewsCleaning(review_filename=negative_reviews_filename)

    positive_reviews = positive_reviews_cleaner.get_reviews()
    negative_reviews = negative_reviews_cleaner.get_reviews()

    positive_labels = [1] * len(positive_reviews)
    negative_labels = [0] * len(negative_reviews)

    reviews = positive_reviews + negative_reviews
    labels = positive_labels + negative_labels

    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.1,
                                                        random_state=1729)

    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1,
                                                      random_state=1729)

    with open(f"{output_folder}/{file_prefix}.train", "w") as fp:
        for line, label in zip(X_train, y_train):
            # \042 is for double quotes
            # f string does not allow you to escape double quotes
            fp.write(f"\042{line}\042###{label}")
            fp.write("\n")

    with open(f"{output_folder}/{file_prefix}.dev", "w") as fp:
        for line, label in zip(X_dev, y_dev):
            # \042 is for double quotes
            # f string does not allow you to escape double quotes
            fp.write(f"\042{line}\042###{label}")
            fp.write("\n")

    with open(f"{output_folder}/{file_prefix}.test", "w") as fp:
        for line, label in zip(X_test, y_test):
            # \042 is for double quotes
            # f string does not allow you to escape double quotes
            fp.write(f"\042{line}\042###{label}")
            fp.write("\n")

    console.print(f"[green] Wrote {file_prefix}.train .dev and .test splits")


if __name__ == "__main__":
    form_blitzer_datasets()
