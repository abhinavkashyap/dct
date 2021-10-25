from dct.evaluation.sim.sim_models import WordAveraging
from dct.evaluation.sim.sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer
import sentencepiece as spm
import torch
from typing import List


class SIM:
    def __init__(
        self, similarity_model: str, sentencepiece_model: str, run_on_cpu=False
    ):
        """

        Parameters
        ----------
        similarity_model: str
            The similarity model filepath
        sentencepiece_model: str
            The sentencepiece model filepath
        run_on_cpu: bool
            Whether to the run the model on the cpu (NO GPU)
        """
        self.tok = TreebankWordTokenizer()
        self.run_on_cpu = run_on_cpu
        model = torch.load(similarity_model, map_location="cpu")
        state_dict = model["state_dict"]
        vocab_words = model["vocab_words"]
        args = model["args"]
        # turn off gpu
        self.model = WordAveraging(args, vocab_words)
        self.model.load_state_dict(state_dict, strict=True)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sentencepiece_model)
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available() and not self.run_on_cpu
            else torch.device("cpu")
        )

        self.model.to(self.device)
        self.model.eval()

    def make_example(self, sentence, model):
        sentence = sentence.lower()
        sentence = " ".join(self.tok.tokenize(sentence))
        sentence = self.sp.EncodeAsPieces(sentence)
        wp1 = Example(" ".join(sentence))
        wp1.populate_embeddings(model.vocab)
        return wp1

    def find_similarity(self, batch1: List[str], batch2: List[str]):
        """Finds similarity betweeen two batches of sentences
        That is [cos(batch1[0], batch2[0]), cos(batch1[1], batch2[1]), ...]

        Parameters
        ----------
        batch1: List[str]
            The first batch of strings
        batch2: List[str]
            The second batch of strings

        Returns
        -------
        List[float]
            Similarities between two batches of sentences
        """
        with torch.no_grad():
            s1 = [self.make_example(sentence, self.model) for sentence in batch1]
            s2 = [self.make_example(sentence, self.model) for sentence in batch2]
            wx1, wl1, wm1 = self.model.torchify_batch(s1, self.device)
            wx2, wl2, wm2 = self.model.torchify_batch(s2, self.device)
            scores = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
            return [x.item() for x in scores]


if __name__ == "__main__":
    sim = SIM()
    print(
        sim.find_similarity(
            batch1=["The world is not good enough", "good"],
            batch2=["The world is not good enough", "na"],
        )
    )
