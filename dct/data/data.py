from typing import List


class InputExample(object):
    """
    Taken from https://github.com/srush/transformers/blob/master/examples/utils_ner.py
    A single training/test example for token classification.
    """

    def __init__(self, guid, words, label):
        """

        Parameters
        ----------
        guid: str
            An identifier for an example
        words: List[str]
            The bag of words in the example
        label: int
            The label for the example
        """
        self.guid: str = guid
        self.words: List[str] = words
        self.label: int = label
