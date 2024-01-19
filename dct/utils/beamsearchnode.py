import torch
from typing import Tuple, Union
from dataclasses import dataclass

# This has been taken from
# https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py


@dataclass(order=True)
class BeamSearchNode:
    def __init__(
        self,
        hiddenstate: Union[
            torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]
        ],
        previousNode,
        wordId,
        logProb: float,
        length: int,
    ):
        """

        Parameters
        ----------
        hiddenstate: torch.FloatTensor
           (1*H) or (1*H, 1*H)
        previousNode: BeamSearcNode
            The previous node in beam search
        wordId
        logProb: float
            Log probability of the path upto the current node
        length: int
            The length of the path upto the current node
        """
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
