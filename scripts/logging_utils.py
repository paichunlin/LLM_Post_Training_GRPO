"""
logging utils for GRPO
"""
import logging
import torch
import wandb
from eval_utils import load_policy_into_vllm_instance, get_LLM_responses, evaluate

logger = logging.getLogger(__name__)

@torch.no_grad()
def logging_eval_metrics(wandb, 
                    policy_model,
                    eval_model,
                    eval_prompts,
                    ground_truths,
                    iter_num,                    
                    args):
    load_policy_into_vllm_instance(policy_model, eval_model)
    responses, _ = get_LLM_responses(eval_model, 
                                     eval_prompts,
                                     args.eval_temperature,
                                     args.eval_num_responses,
                                     args)

    metrics_summary, _ = evaluate(responses, ground_truths, "eval/")
    metrics_summary["iter"] = iter_num
    wandb.log(metrics_summary)

@torch.no_grad()
def logging_train_metrics(wandb,
                          global_step,
                          epoch,
                          step,
                          iter_num,
                          loss,
                          reward_stat,
                          train_stat,                   
                          average_reward,
                          average_response_length,
                          average_entropy):
    metrics_summary = {}
    metrics_summary["global_step"] = global_step
    metrics_summary["epoch"] = epoch
    metrics_summary["step"] = step
    metrics_summary["iter_num"] = iter_num
    metrics_summary["train/loss"] = loss.item()
    metrics_summary["train/batch_mean_reward"] = reward_stat["answer_reward"].item()
    metrics_summary["train/accumulative_mean_reward"] = average_reward
    metrics_summary["train/clip_frac"] = train_stat["clip_frac"]
    metrics_summary["train/average_response_length"] = average_response_length 
    metrics_summary["train/average_entropy"] = average_entropy.item()
    wandb.log(metrics_summary)

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

