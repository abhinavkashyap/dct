import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(
        self,
        emsize: int,
        nhidden: int,
        vocab_size: int,
        nlayers: int,
        dropout: float = 0.0,
        embedding: torch.Tensor = None,
        is_bidirectional: bool = True
    ):
        """

        Parameters
        ----------
        emsize: int
            The embedding size
        nhidden: int
            The hidden size of the encoder
        vocab_size: int
            The vocabulary size
        nlayers: int
            The number of layers of the encoder
        dropout: float
            Dropout to apply if the number of layers
            in the LSTM encoder
        embedding: torch.Tensor
            Pretrained embedding. Optional
            If not given we will train the embedding from scratch
        is_bidirectional: bool
            Bidirectional encoder?
        """

        super(GRUEncoder, self).__init__()
        self.embedding_size = emsize
        self.hidden_size = nhidden
        self.vocab_size = vocab_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.is_bidirectional = is_bidirectional

        if embedding is None:
            self.enc_embedding = nn.Embedding(self.vocab_size, emsize)
        else:
            self.enc_embedding = nn.Embedding.from_pretrained(embeddings=embedding,
                                                              freeze=False)
        self.encoder = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.nlayers,
            dropout=self.dropout,
            bidirectional=is_bidirectional,
            batch_first=True
        )

        self.drop = nn.Dropout(0.5)

        self.init_weights()

    def forward(self, inp: torch.Tensor, lengths: torch.Tensor, pad_idx: int):
        # encoder_outputs: batch_size, max_len, hidden_dimension
        encoder_outputs = self.encode(inp, lengths, pad_idx)

        return encoder_outputs

    def encode(self, inp: torch.Tensor, lengths: torch.Tensor, pad_idx: int):
        """

        Parameters
        ----------
        inp: torch.Tensor
            size: [B, L]
            B - Batch size
            L - Number of tokens
            Padded to L
        lengths: torch.Tensor
            size: [B]
            Real lengths of lines
        pad_idx: int
            The index used for padding

        Returns
        -------
        torch.Tensor
            size: B, H
            B - Batch size
            H - Hidden size

        """
        embs = self.enc_embedding(inp)  # [B, L, E]

        # B * L * H - if unidirectional
        # B * L * 2H - if bidirectional
        lstm_outputs, h_n = self.encoder(embs)

        if self.is_bidirectional:
            B, L, _ = lstm_outputs.size()
            lstm_outputs = lstm_outputs.view(B, L, 2, self.hidden_size)

            # B, L, H
            lstm_first_direction = lstm_outputs[:, :, 0, :]
            lstm_second_direction = lstm_outputs[:, :, 1, :]
            final_lstm_outputs = torch.cat([lstm_first_direction, lstm_second_direction], dim=2)
            final_lstm_outputs = self.drop(final_lstm_outputs)
        else:
            final_lstm_outputs = lstm_outputs

        return final_lstm_outputs

    def init_weights(self):
        initrange = 0.1
        self.enc_embedding.weight.data.uniform_(-initrange, initrange)

        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
