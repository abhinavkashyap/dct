import torch.nn as nn
import torch


class VanillaEmbedder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, reduce=None):
        """A wrapper for vanilla Embedding Module. Returns the
        embedding for every token in the input

        Parameters
        ----------
        vocab_size: int
            The size of the vocab for embedding
        emb_dim: int
            Embedding dimension
        reduce: str
            One of [mean, concat]
            mean: Takes the average of word embeddings of every sentence
            concat: Concatenates word embeddings of every sentence
            mean returns N * D
            concat returns N *TD
        """
        super(VanillaEmbedder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.reduce = reduce

        self.embedder = nn.Embedding(vocab_size, emb_dim)

    def forward(self, inp):
        """

        Parameters
        ----------
        inp: torch.LongTensor
            N * T
            N - batch size
            T - The number of tokens
        Returns
        -------
        torch.Tensor
            N * T * D
            N - batch size
            T - the number of tokens
            D - the embedding dimension
        """
        emb = self.embedder(inp)

        if self.reduce == "mean":
            emb = torch.mean(emb, dim=1)
        elif self.reduce == "concat":
            N, T, D = emb.size()
            emb = emb.view(N, T * D)

        return emb
