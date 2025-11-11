# GRPO Group Relative Policy Optimization
This repo implements Group Relative Policy Optimization from these papers 
    * DeepSeekMath: https://arxiv.org/pdf/2402.03300
    * DeepSeek-R1: https://arxiv.org/abs/2501.12948
    
I also have improved it further based on the following three papers:
* DAPO: https://arxiv.org/pdf/2503.14476
* GRPO+: https://www.together.ai/blog/deepcoder
* DrGRPO: https://arxiv.org/pdf/2503.20783

## Multi-GPU Setup
This GRPO training requires at least two A100 GPUs. One is used for training the model. The other is used for evaluation and rollout.

## Setup
This repo use `uv` to manage dependencies.
1. install uv
```
pip install uv
```

2. Install all packages except `flash-attn`, then install the rest
```
uv sync --no-install-package flash-attn
uv sync
```
3. Run the command 
```
python scripts/grpo_trainer.py \
    --input-path "./data/gsm8k/sft_with_answer.jsonl" \
    --model-name-or-path "Qwen/Qwen2.5-Math-1.5B" \
    --project-name "grpo" \
    --experiment-name "grpo_batch128" \
    --batch-size "2" \
    --gradient-accumulation-steps "8" \
    --output-dir "./Qwen2.5-Math-1.5B/Checkpoint" \
```
