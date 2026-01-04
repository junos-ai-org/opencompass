"""Evaluation config for MMTU benchmark on LLaDA-8B-Instruct and Qwen2-7B-Instruct.

This configuration evaluates both models on the MMTU (Massive Multi-Task Table
Understanding) benchmark using 50 sample questions for quick testing.

Usage:
    python run.py configs/eval_mmtu_llada_qwen2_50.py
"""

from mmengine.config import read_base

with read_base():
    from .datasets.mmtu.mmtu_gen_b510c3 import (
        mmtu_reader_cfg,
        mmtu_infer_cfg,
        mmtu_eval_cfg,
    )

from opencompass.models import LLaDAModel, HuggingFacewithChatTemplate
from opencompass.datasets import MMTUDataset

# Model configurations
models = [
    # LLaDA 8B Instruct - Diffusion-based language model
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct',
        path='GSAI-ML/LLaDA-8B-Instruct',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # LLaDA diffusion parameters
        gen_length=256,
        gen_blocksize=32,
        gen_steps=256,
        diff_confidence_eos_eot_inf=True,
        diff_logits_eos_inf=False,
    ),
    # Qwen2 7B Instruct - Autoregressive transformer model
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2-7b-instruct-hf',
        path='Qwen/Qwen2-7B-Instruct',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

# Reader configuration with 50 samples
mmtu_reader_cfg_50 = dict(
    **mmtu_reader_cfg,
    test_range='[0:50]',  # First 50 questions
)

# Dataset configuration
datasets = [
    dict(
        type=MMTUDataset,
        abbr='mmtu-50',
        path='MMTU-benchmark/MMTU',
        reader_cfg=mmtu_reader_cfg_50,
        infer_cfg=mmtu_infer_cfg,
        eval_cfg=mmtu_eval_cfg,
    ),
]
