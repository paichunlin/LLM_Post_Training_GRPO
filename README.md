### LLM Post Training: Beyond GRPO for hard math reasoning and agentic reasoning
I implemented DeepSeek's GRPO from the ground up and extended it with state-of-art research findings from DAPO and others for hard math reasoning and agentic reasoning.

### Multi-GPU Setup
This GRPO training requires at least two A100 or H100 GPUs. One is used for training the model. The other is used for evaluation and rollout with vLLM.

### Setup
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
