"""LLaDA evaluation on GSM8K with confidence-based decoding.

Configuration:
- gen_length=512, block_size=512 (1 block, fully parallel diffusion)
- diff_confidence_eos_eot_inf=True (confidence-based EOS handling for math)

Usage:
    python run.py examples/eval_llada_gsm8k_len512_block512_confidence.py \
        -w outputs/llada_gsm8k_len512_block512_confidence
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.models.llada.llada_8b_instruct import models

datasets = gsm8k_datasets

# LLaDA diffusion parameters
eval_cfg = dict(
    gen_length=512,
    gen_blocksize=512,
    gen_steps=512,
    batch_size=1,
    batch_size_=1,
    diff_confidence_eos_eot_inf=True,  # Confidence-based decoding for math
    diff_logits_eos_inf=False,
)

for model in models:
    model.update(eval_cfg)
