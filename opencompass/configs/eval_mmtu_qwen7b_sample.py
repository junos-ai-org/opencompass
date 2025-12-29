"""Run config for evaluating Qwen2-7B-Instruct on MMTU with 3 sample questions.

Usage:
    python run.py configs/eval_mmtu_qwen7b_sample.py
"""

from mmengine.config import read_base

with read_base():
    from .datasets.mmtu.mmtu_gen_b510c3 import (
        mmtu_datasets,
        mmtu_reader_cfg,
        mmtu_infer_cfg,
        mmtu_eval_cfg,
    )

from opencompass.models import HuggingFacewithChatTemplate
from opencompass.datasets import MMTUDataset

# Model configuration - Qwen2 7B Instruct
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2-7b-instruct-hf',
        path='Qwen/Qwen2-7B-Instruct',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

# Override reader_cfg to only use 3 samples
mmtu_reader_cfg_sample = dict(
    **mmtu_reader_cfg,
    test_range='[0:3]',  # Only first 3 questions
)

# Dataset configuration with sampling
datasets = [
    dict(
        type=MMTUDataset,
        abbr='mmtu-sample3',
        path='MMTU-benchmark/MMTU',
        reader_cfg=mmtu_reader_cfg_sample,
        infer_cfg=mmtu_infer_cfg,
        eval_cfg=mmtu_eval_cfg,
    ),
]
