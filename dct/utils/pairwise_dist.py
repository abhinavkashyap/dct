import torch


def pairwise_cos(
    first_batch: torch.Tensor, second_batch: torch.Tensor, normalize: bool = True
):
    """

    Parameters
    ----------
    first_batch: torch.Tensor
        M, H
        M - number of points
        H - feature dimension
    second_batch: torch.Tensor
        N, H
        N - number of points
        H - feature dimensions
    normalize: bool
        Whether to normalize the tensors
        The last dimension will be used to normalize

    Returns
    -------
    torch.Tensor
        M, N
        pairwise distance between two sets of objects
    """

    if normalize:
        first_batch = first_batch / first_batch.norm(dim=-1, p=2, keepdim=True)
        second_batch = second_batch / second_batch.norm(dim=-1, p=2, keepdim=True)

    # for every target data, get the similarity with all other src data in the batch
    cos_sim = torch.mm(first_batch, second_batch.transpose(0, 1))

    return cos_sim


def pairwise_attr_intersection(
    first_batch: torch.Tensor, second_batch: torch.Tensor, device=torch.device("cpu")
):
    """Calculates the  attribute intersection between
    every data pint in first_batch with every data point in the second one
    [[1, 2, 3]           [[1, 2, 3]]
                    and
    [4, 5, 6]]
    ==========
    [3, 0]
    since the first object in first_batch matches with all the three in the second one
    and the second object in the first batch matches with none on the second batch tensor

    Note that the attributes are assumed to be arranged in a particular order
    That is, every column (dimension) corresponds to some particular attribute
    This is usually the case for our application. Not generally applicable
    for all use cases

    Parameters
    ----------
    first_batch: torch.Tensor
        M, H
        M - Number of data points in the first batch
        H - The feature size

    second_batch: torch.Tensor
        N, H
        N - Number of points in the second batch
        H - The feature size

    Returns
    -------
    M, N
        The pairwise intersection of attributes between
        the two batch of tensors
    """
    first_batch_sz = first_batch.size(0)
    second_batch_sz = second_batch.size(0)

    # repeat every row of the first batch
    # to match the number of second batch number of rows
    repeats = torch.LongTensor([second_batch_sz] * first_batch_sz).to(device)
    # first_batch_sz * trg_batch_sz, H
    first_batch_rep = first_batch.repeat_interleave(repeats=repeats, dim=0)

    # repeat the second batch matrix first_batch_siz number of times
    # first_batch_sz * trg_batch_sz, H
    second_batch_rep = second_batch.repeat(first_batch_sz, 1)

    # take the difference between them
    diff = first_batch_rep - second_batch_rep

    diff = (diff == 0).sum(dim=1)

    diff = diff.reshape(first_batch_sz, second_batch_sz)

    return diff
