"""
data utils for GRPO
"""
import json
from xopen import xopen
import torch
from torch import Tensor
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

def get_prompts(input_examples):
    return [example["prompt"] for example in input_examples]

def get_responses_from_data(input_examples):
    return [example["response"] for example in input_examples]

def get_answers_from_data(input_examples):
    return [example["answer"] for example in input_examples]

def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    enable_prompt_mask,
    device = None,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    question_dict = tokenizer(prompt_strs)
    question_tokens_list = question_dict["input_ids"]
    response_dict = tokenizer(output_strs)
    response_tokens_list = response_dict["input_ids"]
    max_prompt_and_output_lens = max([len(question_tokens + response_tokens) for question_tokens, response_tokens in zip(question_tokens_list, response_tokens_list)])
    input_ids = []
    labels = []
    response_mask = []
    padding_token_id = tokenizer.pad_token_id
    total_response_length = 0.0
    total_batch_size = len(question_tokens_list)

    for question_tokens, response_tokens in zip(question_tokens_list, response_tokens_list):
      len_question_tokens = len(question_tokens)
      len_response_tokens = len(response_tokens)
      total_response_length += len_response_tokens
      padding_length = max_prompt_and_output_lens - len_question_tokens - len_response_tokens
      combine = question_tokens + response_tokens + [padding_token_id]*padding_length
      input_ids.append(combine[:-1])
      labels.append(combine[1:])
      if enable_prompt_mask:
        response_mask.append([0]*(len_question_tokens - 1)  + [1]*len_response_tokens + [0]*padding_length)
      else:
        response_mask.append([1]*(len_question_tokens - 1)  + [1]*len_response_tokens + [0]*padding_length)

    out_dict = {"input_ids":torch.Tensor(input_ids).to(dtype=torch.long, device=device),
                "labels":torch.Tensor(labels).to(dtype=torch.long, device=device),
                "response_mask":torch.Tensor(response_mask).to(dtype=torch.bool, device=device).requires_grad_(requires_grad=False),
                "average_response_length": total_response_length/total_batch_size}
    return out_dict

def get_input_examples(input_path):
    input_examples = []
    with xopen(input_path) as f:
        for line in f:
            input_examples.append(json.loads(line))
    return input_examples

def get_input_examples(input_path):
    input_examples = []
    with xopen(input_path) as f:
        for line in f:
            input_examples.append(json.loads(line))
    return input_examples

def get_answers(input_examples):
    return [example["answer"] for example in input_examples]
