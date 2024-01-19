import torch
from torch import nn
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List
from dct.utils.beamsearchnode import BeamSearchNode
from queue import PriorityQueue
import operator
from transformers.generation_utils import top_k_top_p_filtering


class Seq2Seq(nn.Module):
    def __init__(
        self,
        emsize,
        nhidden,
        ntokens,
        nlayers,
        noise_r=0.2,
        dropout=0,
        tie_enc_dec_embedding=False,
        encoder=None,
        gen_greedy=True,
    ):
        super().__init__()
        self._noise_r = noise_r
        self.internal_repr_size = nhidden
        self._device = None
        self.tie_enc_dec_embedding = tie_enc_dec_embedding
        self.enc_embedding = nn.Embedding(ntokens, emsize)
        self.dec_embedding = nn.Embedding(ntokens, emsize)
        self.encoder = encoder
        self.gen_greedy = gen_greedy

        if self.tie_enc_dec_embedding:
            self.dec_embedding.weight = self.enc_embedding.weight

        if encoder is None:
            self.encoder = nn.LSTM(
                input_size=emsize,
                hidden_size=nhidden,
                num_layers=nlayers,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.encoder = encoder

        self.decoder = nn.LSTM(
            input_size=emsize + nhidden,
            hidden_size=nhidden,
            num_layers=1,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(nhidden, ntokens)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def noise_anneal(self, alpha):
        self._noise_r *= alpha

    @classmethod
    def from_opts(cls, opts):
        return cls(
            emsize=opts["ae_emb_size"],
            nhidden=opts["ae_hidden_size"],
            ntokens=opts["vocab_size"],
            nlayers=opts["ae_n_layers"],
            noise_r=opts["noise_r"],
            dropout=opts["ae_dropout"],
            tie_enc_dec_embedding=opts.get("tie_enc_dec_embedding", False),
            encoder=opts["encoder"],
            gen_greedy=opts["gen_greedy"],
        )

    def init_weights(self):
        initrange = 0.1
        self.enc_embedding.weight.data.uniform_(-initrange, initrange)
        self.dec_embedding.weight.data.uniform_(-initrange, initrange)

        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def forward(
        self,
        inp,
        dec_inp,
        lengths,
        pad_idx,
        noise=False,
    ):

        # encoder_outputs: batch_size, max_len, hidden_dimension
        # encoder_hn = batch_size, hidden_size
        # encoder_cn = batch_size, hidden_size
        encoder_outputs, (encoder_hn, encoder_cn) = self.encode(
            inp, lengths, noise, pad_idx
        )  # [B, H]

        # decoder_logits: B, L, V
        # decoder_hn = 1, batch_size, hidden_size
        # decoder_cn = 1, batch_size, hidden_size
        decoder_logits, (decoder_hn, decoder_cn) = self.decode((encoder_hn, encoder_cn), dec_inp)

        return (
            encoder_outputs,
            (encoder_hn, encoder_cn),
            decoder_logits,
            (decoder_hn, decoder_cn),
        )

    def encode(self, inp, lengths, noise, pad_idx):
        """Encoding internal representation of inputs
        :param inp: input tokens, size: [B, L]
        :param lengths: real lengths of lines, size: [B]
        :param noise: should add noise for out representation
        :param pad_idx
        :return: internal representation of inp, size: [B, H]
        """
        embs = self.enc_embedding(inp)  # [B, L, E]
        packed_embeddings = pack_padded_sequence(
            embs, lengths=lengths, batch_first=True, enforce_sorted=False
        )

        # lstm_outputs: batch_size, seqlen, hidden_dimension
        lstm_outputs, (h_n, c_n) = self.encoder(packed_embeddings)
        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True, padding_value=pad_idx)

        h_n = h_n[-1]  # getting last layer, size: [B, H]
        h_n = h_n / torch.norm(h_n, p=2, dim=1, keepdim=True)

        c_n = c_n[-1]
        c_n = c_n / torch.norm(c_n, p=2, dim=1, keepdim=True)

        if noise:
            assert self._noise_r > 0
            gauss_noise = torch.normal(mean=torch.zeros_like(h_n), std=self._noise_r)
            h_n += gauss_noise
            c_n += gauss_noise

        return lstm_outputs, (h_n, c_n)

    def decode(self, decoder_init, inp):
        """Decoding an internal representation into tokens probs
        :param decoder_init: internal representation of inp tokens, size: [B, H]
        :param inp: inp tokens in batch, size: [B, L]
        :return: unsoftmaxed probs
        """

        maxlen = inp.size(1)

        # batch_size, hidden_size
        encoder_hn, encoder_cn = decoder_init

        assert encoder_hn.ndimension() == 2
        assert encoder_cn.ndimension() == 2
        assert inp.ndimension() == 2
        assert encoder_hn.size(0) == inp.size(
            0
        ), f"Init hidden and input size should match {encoder_hn.size()} - {inp.size()}"

        # copy the internal representation for every token in line
        hiddens = encoder_hn.unsqueeze(1).repeat(1, maxlen, 1)  # size: [B, L, H]

        embs = self.dec_embedding(inp)  # size: [B, L, E2]
        augmented_embs = torch.cat([embs, hiddens], -1)  # concat, size: [B, L, E2+H]

        decoder_init = encoder_hn.unsqueeze(0), encoder_cn.unsqueeze(0)

        decoder_lstm_output, decoder_state = self.decoder(augmented_embs, decoder_init)
        decoder_hn, decoder_cn = decoder_state
        decoder_hn = decoder_hn.squeeze(0)
        decoder_cn = decoder_cn.squeeze(0)

        decoder_state = (decoder_hn, decoder_cn)

        decoder_logits = self.linear(decoder_lstm_output.contiguous())  # size: [B, L, V]
        return decoder_logits, decoder_state

    @staticmethod
    def sample_next_idxs(
        inp, greedy=True, nucleus=False, top_k: int = 0, top_p=1.0, min_tokens_to_keep=1
    ):
        """Sample next idxs
        :param inp: inp tokens logits(unnormalized probs), size: [B, V]
        :param greedy: choose with the max prob
        :param nucleus: set True for nucleus sampling

        :return: tokens idxs, size: [B]
        """
        logits = inp
        if greedy:
            assert nucleus is False
            return torch.argmax(softmax(logits, dim=-1), -1)
        if nucleus:
            assert greedy is False
            logits = top_k_top_p_filtering(
                inp, top_k=top_k, top_p=top_p, min_tokens_to_keep=min_tokens_to_keep
            )

        probs = softmax(logits, dim=-1)
        return probs.multinomial(num_samples=1).squeeze(-1)

    def generate(
        self,
        encoder_outputs,
        encoder_state,
        internal_repr,
        sos_idx,
        eos_idx,
        maxlen,
        gen_greedy=True,
        nucleus_sampling=False,
        top_k=1,
        top_p: float = 1.0,
        min_tokens_to_keep=1,
        temperature=1.0
    ):
        if gen_greedy:
            return self.generate_greedy(
                encoder_outputs,
                encoder_state,
                internal_repr,
                sos_idx,
                maxlen,
                greedy=True,
            )
        elif nucleus_sampling:
            return self.nucleus_sampling(
                encoder_outputs=encoder_outputs,
                encoder_state=encoder_state,
                internal_repr=internal_repr,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                maxlen=maxlen,
                top_k=top_k,
                top_p=top_p,
                min_tokens_to_keep=min_tokens_to_keep,
                temperature=temperature
            )

        else:
            return self.generate_beam(
                encoder_outputs,
                encoder_state,
                internal_repr,
                sos_idx,
                eos_idx,
                maxlen,
            )

    def generate_greedy(
        self,
        encoder_outputs,
        encoder_state,
        internal_repr,
        sos_idx,
        maxlen,
        greedy,
    ):
        """

        :param encoder_state: torch.Tensor
        :param internal_repr: torch.Tensor
            batch_size * hidden_size
        :param sos_idx:
        :param maxlen:
        :param greedy:
        :return:
        """
        batch_size = internal_repr.size(0)

        # prepare a tensor for generated idxs
        generated_idxs = torch.zeros(maxlen, batch_size, dtype=torch.long).to(self.device)  # [L, B]
        # set SOS as first token
        generated_idxs[0] = sos_idx

        encoder_hn, encoder_cn = encoder_state
        encoder_hn = encoder_hn.unsqueeze(0)
        encoder_cn = encoder_cn.unsqueeze(0)
        state = (encoder_hn, encoder_cn)

        for token_idx in range(maxlen - 1):
            cur_tokens = generated_idxs[token_idx].unsqueeze(1)  # [B, 1]

            cur_embs = self.dec_embedding(cur_tokens)  # [B, 1, E2]
            inputs = torch.cat([cur_embs, internal_repr.unsqueeze(1)], -1)  # [B, 1, E2+H]

            output, state = self.decoder(inputs, state)  # output size: [B, 1, H]
            decoded = self.linear(output.squeeze(1))  # [B, V]
            generated_idxs[token_idx + 1] = self.sample_next_idxs(decoded, greedy)  # [B]

        return generated_idxs.transpose(0, 1)

    def generate_beam(
        self,
        encoder_outputs: torch.Tensor,
        encoder_state: torch.Tensor,
        internal_repr: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        maxlen: int,
    ) -> (List[List[int]], torch.FloatTensor):
        """

        Parameters
        ----------
        encoder_outputs: torch.Tensor
        encoder_state: Tuple[torch.Tensor, torch.Tensor]
            size: batch_size, maxlen, hidden_size
        internal_repr: torch.Tensor
        sos_idx: int
            The start index of the decoder
        eos_idx: int
            End of sentence idx
        maxlen: int
            Maximum length to generate for

        Returns
        -------

        """

        # N * T * src_hidden_dim
        # batch_size * hidden_dim
        # batch_size * hidden_dim
        batch_size = internal_repr.size(0)
        beam_width = 5
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        batch_scores = []

        decoder_hiddens = encoder_state

        #####################################################################################
        #                      DECODING STARTS SENTENCE BY SENTENCE
        #####################################################################################
        for idx in range(batch_size):

            decoder_hidden = (
                decoder_hiddens[0][idx, :].unsqueeze(0),  # 1 * H
                decoder_hiddens[1][idx, :].unsqueeze(0),  # 1 * H
            )

            # 1 * T * H
            encoder_output: torch.Tensor = encoder_outputs[idx, :, :].unsqueeze(0)
            assert encoder_output.ndimension() == 3
            assert encoder_output.size(0) == 1

            # Start with the start of the sentence token
            # decoder_input - size: 1 * 1
            decoder_input: torch.LongTensor = torch.LongTensor([[sos_idx]]).view(-1, 1)

            decoder_input = decoder_input.to(self.device)
            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            # noinspection PyTypeChecker
            node = BeamSearchNode(
                hiddenstate=decoder_hidden,
                previousNode=None,
                wordId=decoder_input,
                logProb=0,
                length=1,
            )

            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000:
                    break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == eos_idx and n.prevNode is not None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decoder_logit - 1 * 1 * V
                # decoder_probs - 1 * 1 * V
                # decoder_output - 1 * 1 * trg_hidden-dimension
                # decoder_hidden - (h_t, c_t) - (1*trg_hidden_dimension, 1*trg_hidden_dimension)
                (decoder_logits, decoder_hidden) = self.decode(decoder_hidden, decoder_input)

                decoder_probs = self.softmax(decoder_logits)
                # PUT HERE REAL BEAM SEARCH OF TOP
                probs, indexes = torch.topk(decoder_probs, beam_width)
                log_prob = torch.log(probs)
                log_prob = log_prob.squeeze(0)
                indexes = indexes.squeeze(0)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(
                        hiddenstate=decoder_hidden,
                        previousNode=n,
                        wordId=decoded_t,
                        logProb=n.logp + log_p,
                        length=n.leng + 1,
                    )
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nextnode = nextnodes[i]
                    nodes.put((score, nextnode))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances: List[int] = []
            scores: List[float] = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = [n.wordid.item()]
                scores.append(pow(2, score))
                # back trace
                while n.prevNode is not None:
                    n = n.prevNode
                    utterance.append(n.wordid.item())

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.extend(utterances)
            batch_scores.extend(scores)

        avg_perplexity = sum(batch_scores) / len(batch_scores)
        avg_perplexity = torch.FloatTensor([avg_perplexity])
        return decoded_batch

    def nucleus_sampling(
        self,
        encoder_outputs: torch.Tensor,
        encoder_state: torch.Tensor,
        internal_repr: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        maxlen: int,
        top_k: int = 0,
        top_p: float = 1.0,
        min_tokens_to_keep=1,
        temperature:float=1.0
    ):
        batch_size = internal_repr.size(0)
        # prepare a tensor for generated idxs
        generated_idxs = torch.zeros(maxlen, batch_size, dtype=torch.long).to(self.device)  # [L, B]
        # set SOS as first token
        generated_idxs[0] = sos_idx

        encoder_hn, encoder_cn = encoder_state
        encoder_hn = encoder_hn.unsqueeze(0)
        encoder_cn = encoder_cn.unsqueeze(0)
        state = (encoder_hn, encoder_cn)

        for token_idx in range(maxlen - 1):
            cur_tokens = generated_idxs[token_idx].unsqueeze(1)  # [B, 1]

            cur_embs = self.dec_embedding(cur_tokens)  # [B, 1, E2]
            inputs = torch.cat([cur_embs, internal_repr.unsqueeze(1)], -1)  # [B, 1, E2+H]

            output, state = self.decoder(inputs, state)  # output size: [B, 1, H]
            logits = self.linear(output.squeeze(1))  # [B, V]
            logits = logits / temperature
            generated_idxs[token_idx + 1] = self.sample_next_idxs(
                logits,
                nucleus=True,
                top_k=top_k,
                top_p=top_p,
                min_tokens_to_keep=min_tokens_to_keep,
                greedy=False
            )  # [B]

        return generated_idxs.transpose(0, 1)

    @property
    def device(self):
        # lazy insta
        if self._device:
            return self._device
        is_cuda = next(self.parameters()).is_cuda
        self._device = torch.device("cuda" if is_cuda else "cpu")
        return self._device
