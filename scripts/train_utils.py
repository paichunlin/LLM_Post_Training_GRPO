"""
training utils for GRPO
"""
import argparse
import json
import logging
import sys
from statistics import mean
from unittest.mock import patch

from tqdm import tqdm
from transformers import AutoModelForCausalLM

from typing import Callable, Literal

import torch
from torch import Tensor
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
import wandb
from eval_utils import compute_entropy, print_information
from logging_utils import logging_train_metrics

logger = logging.getLogger(__name__)

def init_model(model_id: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
          model_id,
          torch_dtype=torch.bfloat16,
          attn_implementation="flash_attention_2",
        )
    model.to(device)
    return model

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor
):
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    batch, seq_len = input_ids.shape
    labels = labels.to(device)
    model.to(device)
    output = model(input_ids)
    logits = output.logits
    probs = nn.Softmax(dim=-1)(logits)
    probs_shaped = probs.view(-1, probs.shape[-1])
    labels = labels.view(-1)
    probs_for_labels = torch.gather(probs_shaped, 1, labels.unsqueeze(1))
    probs_for_labels = probs_for_labels.view(batch, seq_len)
    log_probs_for_labels = torch.log(probs_for_labels)

    return {
        "log_probs": log_probs_for_labels,
        "probs": probs_for_labels
    }

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

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses,
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    raw_rewards = torch.Tensor([reward_fn(rollout_response, ground_truth)["answer_reward"] for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths)])
    def compute_deltas(rewards: torch.Tensor,
                       normalize_by_std: bool,
                       advantage_eps: float) -> torch.Tensor:
      if normalize_by_std:
        mean_rewards = rewards.mean(dim=-1, keepdim=True)
        std_rewards = rewards.std(dim=-1, keepdim=True)
        normalized_rewards = (rewards - mean_rewards)/(std_rewards + advantage_eps)
        return normalized_rewards
      else:
        mean_rewards = rewards.mean(dim=-1, keepdim=True)
        centered_rewards = rewards - mean_rewards
        return centered_rewards

    group_normalized_rewards = torch.empty_like(raw_rewards)
    for group_id in range(group_size):
      start_idx = group_id * group_size
      end_idx = start_idx + group_size
      group_normalized_rewards[start_idx:end_idx] = compute_deltas(raw_rewards[start_idx:end_idx], normalize_by_std, advantage_eps)

    return group_normalized_rewards, raw_rewards, {"answer_reward": raw_rewards.mean()}

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

        Args:
            raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
                the raw rewards or advantages for each rollout response.
            policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
                the log-probs of the policy.

        Returns:
            torch.Tensor of shape (batch_size, sequence_length):
                the policy gradient per-token loss.
    """
    return -raw_rewards_or_advantages*policy_log_probs

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

        Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

        Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length):
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss
                (used to compute clip fraction).
    """
    unclipped_ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(unclipped_ratios, min=1 - cliprange, max=1 + cliprange)
    clipped_pg_losses = clipped_ratios*advantages
    unclipped_pg_losses = unclipped_ratios*advantages
    
    with torch.no_grad():
        clip_frac = ((clipped_pg_losses < unclipped_pg_losses).float() * response_mask).sum() / response_mask.sum()

    loss = -torch.minimum(unclipped_pg_losses, clipped_pg_losses)
    return loss, {"clip_frac": clip_frac}

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """

    assert loss_type in ["no_baseline","reinforce_with_baseline", "grpo_clip"], "invalid loss type"
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "no_baseline loss type requires raw_rewards"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "reinforce_with_baseline loss type requires advantages"
        advantages = advantages.unsqueeze(1)    
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        assert all(p is not None for p in [advantages, old_log_probs, cliprange]), "grpo_clip loss type requires advantages, old_log_probs, cliprange"
        advantages = advantages.unsqueeze(1)    
        loss, stat = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange, response_mask)
        return loss, stat

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    if dim is None:
      total_nonmask_size = (mask == True).sum()
      tensor = tensor*mask/total_nonmask_size
      mean = tensor.sum()
      return mean
    else:
      nonmask_size_for_dim_axis = (mask == True).sum(dim=dim, keepdim=True)
      tensor = tensor*mask/nonmask_size_for_dim_axis
      mean = tensor.sum(dim=dim, keepdim=True).squeeze()
      return mean


def get_grpo_microbatch_train_loss(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_penalty: float,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio.
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """
    batch_loss, stat = compute_policy_gradient_loss(policy_log_probs,
                                        loss_type,
                                        raw_rewards,
                                        advantages,
                                        old_log_probs,
                                        cliprange,
                                        response_mask)
    if kl_penalty != 0.0:
        loss += kl_penalty * compute_kl_penalty(log_probs=policy_log_probs, ref_log_probs=ref_log_probs)

    loss = torch.mean(masked_mean(batch_loss, response_mask, -1))

    loss.div_(gradient_accumulation_steps)
    return loss, stat

def compute_kl_penalty(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute an estimate of KL(model | ref_model), where the models are given by:
        log_probs [batch trial pos vocab]
        ref_log_probs [batch trial pos vocab]
    Use the estimate:
        KL(p || q) = E_p[q/p - log(q/p) - 1]
    """
    return (torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1).sum(dim=-1).mean()

def process_step(buffered_inputs, 
                 policy_model, 
                 global_step, 
                 epoch, 
                 step, 
                 average_reward,
                 args):
   
   for batch in buffered_inputs:
        iter_num = batch["iter_num"]
        old_log_probs, old_probs, ref_log_probs = batch["old_log_probs"], batch["old_probs"], batch["ref_log_probs"]
        rollout_tokenizer_response, average_response_length = batch["rollout_tokenizer_response"], batch["average_response_length"],
        advantages, raw_rewards, reward_stat = batch["advantages"], batch["raw_rewards"], batch["reward_stat"]
        prompts, responses = batch["prompts"], batch["responses"]

        policy_log_probs_response = get_response_log_probs(policy_model,
                                                           rollout_tokenizer_response["input_ids"],
                                                           rollout_tokenizer_response["labels"])
        policy_log_probs = policy_log_probs_response["log_probs"]
        policy_probs = policy_log_probs_response["probs"]
        entropy_response = compute_entropy(policy_probs, policy_log_probs, rollout_tokenizer_response["response_mask"])

        loss, train_stat = get_grpo_microbatch_train_loss(policy_log_probs,
                                              ref_log_probs,
                                              args.kl_penalty,
                                              rollout_tokenizer_response["response_mask"],
                                              args.gradient_accumulation_steps,
                                              args.loss_type,
                                              raw_rewards,
                                              advantages,
                                              old_log_probs,
                                              args.clip_range)
                       
        print_information(epoch=epoch, 
                          step=step, 
                          loss=loss, 
                          prompts=prompts,
                          rewards=raw_rewards,
                          entropies=entropy_response["entropy"],
                          responses=responses, 
                          deltas=advantages,
                          group_size=args.rollout_num_responses)  

        logging_train_metrics(wandb,
                              global_step,
                              epoch,
                              step,
                              iter_num,
                              loss,
                              reward_stat,
                              train_stat,
                              average_reward,
                              average_response_length,
                              entropy_response["average_entropy"])          
        loss.backward()
