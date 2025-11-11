"""
GRPO for Qwen2.5-Math-1.5B using sft_with_answer.jsonl.
It implements Algorithm 1 from the paper https://arxiv.org/pdf/2402.03300

Algorithm 1 Iterative Group Relative Policy Optimization(GRPO)
Input initial policy model πθinit ; reward function R; task questions D
1: policy model πθ ←πθinit
2: for step = 1, …, n_grpo_steps do
3:     Sample a batch of questions Db from D
4:     Set the old policy model πθold ←πθ
5:     Sample G outputs {o(i)}G_i=1 ∼πθold (·|q) for each question q ∈Db
6:     Compute rewards {r(i)}G_i=1 for each sampled output o(i) by running reward function R(q, o(i))
7:     Compute A(i) with group normalization
8:     for train step = 1, …, n_train_steps_per_rollout_batch do
9:         Update the policy model πθ by maximizing the GRPO-Clip objective (to be discussed, Eq. 21 from the paper)
10:    end for
11: end for
Output πθ

Example Command:

python scripts/grpo_trainer.py \
    --input-path "./data/gsm8k/sft_with_answer.jsonl" \
    --model-name-or-path "Qwen/Qwen2.5-Math-1.5B" \
    --project-name "grpo" \
    --experiment-name "grpo_batch128" \
    --batch-size "2" \
    --gradient-accumulation-steps "8" \
    --output-dir "./Qwen2.5-Math-1.5B/Checkpoint" \

"""
import argparse
import logging
import sys
from unittest.mock import patch

from tqdm import tqdm
from transformers import AutoTokenizer
from grpo_alignment.drgrpo_grader import r1_zero_reward_fn
from torch.utils.data import Dataset, DataLoader
from data_utils import get_input_examples, get_prompts, get_answers, get_responses_from_data, get_answers_from_data, run_tokenize_prompt_and_output
from train_utils import init_model, compute_group_normalized_rewards, get_response_log_probs, process_step
from eval_utils import init_vllm, load_policy_into_vllm_instance, get_LLM_responses
from logging_utils import logging_eval_metrics

import torch
from torch.utils.data import Dataset

import wandb

logger = logging.getLogger(__name__)

class SFTDataset(Dataset):
    def __init__(self, answers, prompts):
        self.answers = answers
        self.prompts = prompts

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return self.answers[idx], self.prompts[idx]

def main(args):
    torch.manual_seed(args.seed)
    # it does not use the tokenizer here, but 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.save_pretrained(save_directory=args.output_dir)
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    assert(num_gpus == 2)

    # Assign GPU 0 for training
    train_device = torch.device("cuda:0")

    # Assign GPU 1 for evaluation
    eval_device = torch.device("cuda:1")

    # Data Preparation: Eval dataset
    eval_input_examples = get_input_examples(args.eval_input_path)
    eval_prompts = get_prompts(eval_input_examples)
    ground_truths = get_answers(eval_input_examples)

    # Data Preparation: SFT dataset D
    train_input_examples = get_input_examples(args.train_input_path)   
    train_prompts = get_prompts(train_input_examples)
    train_responses = get_responses_from_data(train_input_examples)
    train_answers = get_answers_from_data(train_input_examples)
    train_tokenizer_response = run_tokenize_prompt_and_output(train_prompts, train_responses, tokenizer,args.enable_prompt_mask, train_device)

    # Initialize
    # 1. policy model πθinit
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model = init_model(args.model_name_or_path, train_device)
    eval_model = init_vllm(args.model_name_or_path, eval_device, args.seed)
        
    # 2. data loader
    train_ds = SFTDataset(train_answers,
                          train_prompts)

    train_loader =  DataLoader(
                        dataset=train_ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=0)
    
    # 3. optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    config = {
        'batch_size': args.batch_size,
        'grad_accumulation_steps': args.gradient_accumulation_steps,
    }
    wandb.init(project=args.project_name, name=args.experiment_name, config=config)
    total_reward = 0.0
    total_sample = 0
    buffered_inputs = [None for _ in range(args.gradient_accumulation_steps)]
    ref_log_probs = None

    for epoch in tqdm(range(args.n_grpo_steps), desc="GRPO steps"):
        # If using KL penalty, need to get the reference model (freeze it every few epochs)
        if args.kl_penalty != 0.0:
            if epoch % args.compute_ref_model_period == 0:
                ref_model = policy_model.clone()

        for iter_num, (answers, prompts) in tqdm(enumerate(train_loader)):
            logger.info("processing iteration %i", iter_num)

            # Sample responses and evaluate their rewards
            load_policy_into_vllm_instance(policy_model, eval_model)
            rollout_responses, repeat_prompts = get_LLM_responses(eval_model, 
                                                                  prompts,
                                                                  args.rollout_temperature,
                                                                  args.rollout_num_responses,
                                                                  args)
            repeated_ground_truths = [item for item in answers for _ in range(args.rollout_num_responses)]
            advantages, raw_rewards, reward_stat = compute_group_normalized_rewards(r1_zero_reward_fn,
                                                                                    rollout_responses,
                                                                                    repeated_ground_truths,
                                                                                    args.rollout_num_responses,
                                                                                    args.advantage_eps,
                                                                                    args.normalize_by_std)
            with torch.no_grad():
                total_reward += sum(raw_rewards.tolist())
                total_sample += len(raw_rewards.tolist())
        
            rollout_tokenizer_response = run_tokenize_prompt_and_output(repeat_prompts, 
                                                                        rollout_responses, 
                                                                        tokenizer, 
                                                                        args.enable_prompt_mask, 
                                                                        train_device)
        
            if args.kl_penalty != 0.0:
                with torch.no_grad():
                    ref_log_probs_repsonse = get_response_log_probs(ref_model,
                                                           train_tokenizer_response["input_ids"],
                                                           train_tokenizer_response["labels"])
                    ref_log_probs = ref_log_probs_repsonse["log_probs"].to(train_device)


            with torch.no_grad():
                old_log_probs_response = get_response_log_probs(policy_model,
                                                       rollout_tokenizer_response["input_ids"],
                                                       rollout_tokenizer_response["labels"])
                                
                old_log_probs = old_log_probs_response["log_probs"].to(train_device)
                old_probs = old_log_probs_response["probs"].to(train_device)
            
            inputs = {"iter_num": iter_num, 
                      "old_log_probs": old_log_probs,
                      "old_probs": old_probs,
                      "ref_log_probs": ref_log_probs,
                      "rollout_tokenizer_response": rollout_tokenizer_response,
                      "advantages": advantages.to(train_device),
                      "raw_rewards": raw_rewards.to(train_device),
                      "reward_stat": reward_stat,
                      "prompts": repeat_prompts,
                      "responses": rollout_responses, 
                      "average_response_length": rollout_tokenizer_response["average_response_length"]}
            
            buffered_inputs[iter_num % args.gradient_accumulation_steps] = inputs
            if (iter_num + 1) % args.gradient_accumulation_steps == 0:
        
                # Take a number of steps given the responses
                for step in range(args.epochs_per_rollout_batch):
                    with torch.no_grad():
                        global_step = epoch * args.epochs_per_rollout_batch + step
                        average_reward = total_reward/total_sample

                    process_step(buffered_inputs, 
                                 policy_model,
                                 global_step,
                                 epoch, 
                                 step, 
                                 average_reward, 
                                 args)

                    # Backprop and update parameters
                    optimizer.step()
                    optimizer.zero_grad()

                    logging_eval_metrics(wandb, 
                                policy_model,
                                eval_model,
                                eval_prompts,
                                ground_truths,
                                global_step,                    
                                args)
    
    # Output πθ
    policy_model.save_pretrained(save_directory=args.output_dir)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-input-path",
        type=str,
        required=True,
        help="training file input",
    )
    parser.add_argument(
        "--eval-input-path",
        type=str,
        required=True,
        help="eval file input",
    )
    parser.add_argument(
        "--model-name-or-path", help="HF name of the model to use", required=True
    )
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Pojrect name",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to write model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        required=True,
        default="grpo_clip",
        help="loss type: no_baseline, reinforce_with_baseline, grpo_clip",
    )
    parser.add_argument("--seed", help="Seed", type=int, default=64)
    parser.add_argument("--enable-prompt-mask", help="Enable prompt masking", action='store_true')
    parser.add_argument("--normalize-by-std", help="Reward normalized by std?", action='store_true')
    parser.add_argument("--kl-penalty", help="KL penality", type=float, default=0.0)
    parser.add_argument("--n-grpo-steps", help="number of GRPO steps", type=int, default=1)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=8)
    parser.add_argument("--learning-rate", help="Learning rate", type=float, default=0.001)
    parser.add_argument("--gradient-accumulation-steps", help="Gradient accumulation steps", type=int, default=1)
    parser.add_argument("--eval-temperature", help="Sampling temperature", type=float, default=0.0)
    parser.add_argument("--rollout-temperature", help="Sampling temperature", type=float, default=0.7)
    parser.add_argument("--top-p", help="Sampling top_p", type=float, default=1.0)
    parser.add_argument("--min-tokens", help="Sampling min number of tokens", type=int, default=4)
    parser.add_argument("--max-tokens", help="Sampling max number of tokens", type=int, default=300)
    parser.add_argument("--eval-num-responses", help="Sampling number of responses for evaluation", type=int, default=1)
    parser.add_argument("--rollout-num-responses", help="Sampling number of responses for rollout", type=int, default=1)
    parser.add_argument("--compute-ref-model-period", help="Compute ref model period", type=int, default=1)
    parser.add_argument("--epochs-per-rollout-batch", help="Epochs per rollout batch", type=int, default=1)
    parser.add_argument("--advantage-eps", help="epsilon to avoid division by zero during group normalization.", type=float, default=0.000001)
    parser.add_argument("--clip-range", help="epsilon for clip range in GRPO", type=float, default=0.1)

    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))

    main(args)
    logger.info("finished running %s", sys.argv[0])
