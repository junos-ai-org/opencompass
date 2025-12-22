"""LLaDA evaluation on HumanEval with logits-based decoding.

Configuration:
- gen_length=512, block_size=512 (1 block, fully parallel diffusion)
- diff_logits_eos_inf=True (logits-based EOS handling for code generation)

Usage:
    python run.py examples/eval_llada_humaneval_len512_block512_logits.py \
        -w outputs/llada_humaneval_len512_block512_logits
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen import \
        humaneval_datasets
    from opencompass.configs.models.llada.llada_8b_instruct import models

datasets = humaneval_datasets

# LLaDA diffusion parameters
eval_cfg = dict(
    gen_length=512,
    gen_blocksize=512,
    gen_steps=512,
    batch_size=1,
    batch_size_=1,
    diff_confidence_eos_eot_inf=False,
    diff_logits_eos_inf=True,  # Logits-based decoding for code
)

for model in models:
    model.update(eval_cfg)
