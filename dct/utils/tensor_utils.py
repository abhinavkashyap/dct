import torch
from typing import Optional, Dict, List, Union
from loguru import logger


def get_seq_seq_tensors(
    input_text: str,
    max_length: int,
    tok2idx: Dict[str, int],
    unk_idx: int,
    target_text: Optional[str] = None,
    debug: bool = False,
    add_special_tokens: bool = True,
) -> List[torch.LongTensor]:
    """Given an input text and target text, converts them to tensors useful
    for seq-seq kind of modelling

    Parameters
    ----------
    input_text: str
        The text that will serve as input to a sequence to sequence networks
    max_length: int
        The maximum length of input. All the inputs will be padded or truncated
        to this length
    tok2idx: Dict[str, int]
        A mapping from token 2 idx
    unk_idx: int
        The idx for the unknown word in the vocabulary
    target_text: str
        The text that will serve as input to sequence to sequence networks
    debug: bool
        If true, debugging messages are logged
    add_special_tokens: bool
        If True, adds special tokens including padding
        If False, does None of it


    Returns
    -------
    torch.LongTensor, torch.LongTensor, torch.LongTensor
    input_tensor, mask_tensor, output_tensor

    """

    input_text_ = input_text.split()
    input_text_len = len(input_text_)

    if input_text_len == 0:
        input_text_ = ["<empty>"]
        input_text_len = len(input_text_)

    if not add_special_tokens:
        input_text_ = input_text_[: max_length]
        input_text_len = len(input_text_)
        mask_tensor = [1] * len(input_text_)  # 1 indicates to consider the input text.

    if add_special_tokens:
        input_pad_len = max(0, max_length - input_text_len - 2)
        input_text_ = input_text_[: max_length - 2]
        input_text_len = len(input_text_)
        mask_tensor = [1] * len(input_text_)  # 1 indicates to consider the input text.

        input_text_ = ["<s>"] + input_text_ + ["</s>"]
        mask_tensor = [1] + mask_tensor + [1]

        input_text_ += ["<pad>"] * input_pad_len
        mask_tensor += [0] * input_pad_len

    if debug:
        logger.debug(f"encoder_input: {input_text_}")

    input_tensor = [tok2idx.get(token, unk_idx) for token in input_text_]
    input_tensor = torch.LongTensor(input_tensor)
    mask_tensor = torch.LongTensor(mask_tensor)
    input_len_tensor = torch.LongTensor([input_text_len])

    if target_text is not None:
        target_text_ = target_text.split()

        target_text_len = len(target_text_)
        target_pad_len = max(0, max_length - target_text_len - 2)
        target_text_ = target_text_[: max_length - 2]
        target_text_ = ["<s>"] + target_text_ + ["</s>"]
        target_text_ += ["<pad>"] * target_pad_len

        target_input_text = target_text_[:-1]
        target_output_text = target_text_[1:]

        if debug:
            logger.debug(f"decoder_input: {target_input_text}")
            logger.debug(f"decoder_output: {target_output_text}")

        target_input_tensor = [
            tok2idx.get(token, unk_idx) for token in target_input_text
        ]
        target_output_tensor = [
            tok2idx.get(token, unk_idx) for token in target_output_text
        ]

        target_input_tensor = torch.LongTensor(target_input_tensor)
        target_output_tensor = torch.LongTensor(target_output_tensor)

        assert len(target_input_tensor) == max_length - 1, (
            f"len(target_input_tensor): {len(target_input_tensor)}, "
            f"max_length: {max_length}, target text {target_text}"
        )
        assert len(target_output_tensor) == max_length - 1

    else:
        target_input_tensor = torch.LongTensor([])
        target_output_tensor = torch.LongTensor([])

    assert len(input_tensor) == max_length, (
        f"len(input_tensor): {len(input_tensor)}, max_length: {max_length}, "
        f"input text len {input_text_len}"
    )

    return [
        input_tensor,
        input_len_tensor,
        target_input_tensor,
        target_output_tensor,
        mask_tensor,
    ]


def tensors_to_text(
    tensors: Union[torch.LongTensor, List[List[int]]],
    idx2token: Dict[int, str],
    remove_special_tokens: bool = True,
) -> List[str]:
    """Return plain text from tensor. If we encounter a </s> we return
    the sentence till there.

    Parameters
    ----------
    tensors: torch.LongTensor
        A batch of tensors to be converted to text.
        size: batch_size * num_time_steps
    idx2token: Dict[int, str]
        mapping from idx to token
    remove_special_tokens: bool
        Removes special tokens
        If false, we just return the string without removing them
        Useful for debug purposes

    Returns
    -------
    List[str]
        A list of plain text from tensors

    """
    if isinstance(tensors, torch.Tensor):
        decoded_sentences = tensors.tolist()
    else:
        decoded_sentences = tensors

    plain_text_lines = []
    for sentence in decoded_sentences:
        text = []
        for tok_idx in sentence:
            # noinspection PyTypeChecker
            token = idx2token.get(tok_idx)
            if remove_special_tokens:
                if token == "<s>":
                    pass
                elif token == "</s>":
                    break
                elif token == "<pad>":
                    pass
                else:
                    text.append(token)
            else:
                text.append(token)

        text = " ".join(text)
        plain_text_lines.append(text)

    return plain_text_lines


def sample_random_normal_noise(batch_size, internal_repr_size):
    return torch.Tensor(batch_size, internal_repr_size).normal_(0, 1)
