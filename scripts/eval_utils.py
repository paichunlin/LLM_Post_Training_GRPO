"""
evaluation utils for GRPO
"""
from statistics import mean
from unittest.mock import patch

from tqdm import tqdm
from vllm import LLM, SamplingParams
from grpo_alignment.drgrpo_grader import r1_zero_reward_fn

import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed

@torch.no_grad()
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
                        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                        return_value=None
                        )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

@torch.no_grad()
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

@torch.no_grad()
def get_LLM_responses(model, 
                      prompts,
                      temperature,
                      num_responses,
                      args):
    sampling_params = SamplingParams(temperature = temperature, 
                                     top_p = args.top_p, 
                                     max_tokens = args.max_tokens, 
                                     min_tokens = args.min_tokens,
                                     n = num_responses,
                                     seed = args.seed)
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    raw_responses = model.generate(prompts, sampling_params)
    responses = []
    repeat_prompts = []
    for output in tqdm(raw_responses, desc="generating responses"):
        prompt = output.prompt
        for response in output.outputs:
            repeat_prompts.append(prompt)
            responses.append(response.text.strip())
    assert len(responses) == len(prompts)*num_responses
    return responses, repeat_prompts

@torch.no_grad()
def evaluate(responses, ground_truths, prefix="", is_fast_eval=True):
    all_metrics = []
    for response, ground_truth in tqdm(
            zip(responses, ground_truths), desc="evaluating"
        ):
            metrics = r1_zero_reward_fn(response, ground_truth, fast=is_fast_eval)
            all_metrics.append(metrics)

    metrics_summary = {}
    for key in sorted(list(all_metrics[0].keys())):
        metric_value = mean([metrics[key] for metrics in all_metrics])
        logger.info(f"{key}: {metric_value}")
        metrics_summary[prefix + key] = metric_value

    return metrics_summary, all_metrics

@torch.no_grad()
def print_information(epoch: int, 
                      step: int, 
                      loss: torch.Tensor, 
                      prompts: torch.Tensor, 
                      rewards: torch.Tensor,
                      entropies: torch.Tensor,
                      responses: torch.Tensor, 
                      deltas: torch.Tensor,
                      group_size: int):
    print(f"epoch = {epoch}, step = {step}, loss = {loss:.3f}, reward = {rewards.mean():.3f}")
    
    for index, (prompt, response, reward, delta, entropy) in enumerate(zip(prompts, responses, rewards, deltas, entropies)):
        if index == group_size:
            break
            
        print(f"  prompt = {prompt}")
        print(f"  response = {response}, reward = {reward}, delta = {delta:.3f}, entropy = {entropy:.2f}")

@torch.no_grad()
def compute_entropy(policy_probs,
                    policy_log_probs,
                    mask):
    # https://github.com/NVIDIA/NeMo-Aligner/blob/main/nemo_aligner/utils/ppo_utils.py#L52
    # https://github.com/NVIDIA/NeMo-Aligner/blob/main/nemo_aligner/utils/utils.py#L184
    entropy = -torch.sum(policy_probs*policy_log_probs*mask.bool(), dim=-1, keepdim=True).squeeze()
    average_entropy = entropy.mean()
    return {"entropy": entropy,
            "average_entropy": average_entropy}

