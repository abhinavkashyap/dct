import torch.nn as nn
import torch


class ContraLoss(nn.Module):
    def __init__(self, temperature, reduction="sum"):
        """Calculates the contrastive loss for a batch of
        source, positive and negativ batch tensors
        """
        super(ContraLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, source_tensor, pos_batch, neg_batch, do_normalize: bool = False):
        """

        Parameters
        ----------
        source_tensor: torch.Tensor
             B, H
             B - Batch size
             H - Hidden Dimension
        pos_batch: torch.Tensor
            B, P, H
            B - Batch Size
            P - Number of positives
            H - Hidden dimensions
        neg_batch: torch.Tensor
            B, N, H
            B - Batch Size
            N - Number of negatives
            G - Hidden dimension
        do_normalize: bool
            If this is true, then the source tensor
            positive batch and the negatve batch are all normalized
        Returns
        -------
        float
            Contrastive loss
        """

        _, num_positive, _ = pos_batch.size()
        _, num_negative, _ = neg_batch.size()

        if do_normalize:
            source_tensor = source_tensor / torch.norm(
                source_tensor, p=2, dim=1, keepdim=True
            )
            pos_batch = pos_batch / torch.norm(pos_batch, p=2, dim=2, keepdim=True)
            neg_batch = neg_batch / torch.norm(neg_batch, p=2, dim=2, keepdim=True)

        # B * 1 * P
        source_positive = torch.bmm(
            source_tensor.unsqueeze(1), pos_batch.transpose(1, 2)
        )

        # B * P
        source_positive = source_positive.squeeze(1)

        # B * P
        source_positive /= self.temperature

        # B * P
        source_positive = torch.exp(source_positive)

        # B
        source_positive = torch.mean(source_positive, dim=1)

        # B * N
        source_negative = torch.bmm(
            source_tensor.unsqueeze(1), neg_batch.transpose(1, 2)
        )

        # B * N
        source_negative = source_negative.squeeze(1)

        source_negative /= self.temperature

        # B * N
        source_negative = torch.exp(source_negative)

        # B
        source_negative = torch.sum(source_negative, dim=1)

        # divide the positive and the negative scores

        # B
        loss = torch.div(source_positive, source_negative)

        # B
        loss = -torch.log(loss)

        # take the sum of the losses of the batch
        if self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "mean":
            loss = torch.mean(loss)

        else:
            raise ValueError(f"Contrastive loss reduction can be [sum, mean]")
        return loss
