"""LLaDA evaluation on MMLU with basic decoding.

Configuration:
- gen_length=256, block_size=256 (1 block, fully parallel diffusion)
- No special EOS handling (basic decoding)

Usage:
    python run.py examples/eval_llada_mmlu_len256_block256_basic.py \
        -w outputs/llada_mmlu_len256_block256_basic
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.models.llada.llada_8b_instruct import models

datasets = mmlu_datasets

# LLaDA diffusion parameters
eval_cfg = dict(
    gen_length=256,
    gen_blocksize=256,
    gen_steps=256,
    batch_size=1,
    batch_size_=1,
    diff_confidence_eos_eot_inf=False,  # Basic decoding
    diff_logits_eos_inf=False,
)

for model in models:
    model.update(eval_cfg)
