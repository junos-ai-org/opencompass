"""LLaDA evaluation on GSM8K (first 50 examples only) - for testing.

Configuration:
- gen_length=512, block_size=512 (1 block, fully parallel diffusion)
- diff_confidence_eos_eot_inf=True (confidence-based EOS handling for math)
- test_range='[0:50]' (only first 50 examples)

Usage:
    python run.py examples/eval_llada_gsm8k_test50.py \
        -w outputs/llada_gsm8k_test50
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.models.llada.llada_8b_instruct import models

datasets = gsm8k_datasets

# Limit to first 50 examples for testing
for dataset in datasets:
    dataset['reader_cfg']['test_range'] = '[0:50]'

# LLaDA diffusion parameters
eval_cfg = dict(
    gen_length=512,
    gen_blocksize=512,
    gen_steps=512,
    batch_size=1,
    batch_size_=1,
    diff_confidence_eos_eot_inf=True,
    diff_logits_eos_inf=False,
)

for model in models:
    model.update(eval_cfg)
