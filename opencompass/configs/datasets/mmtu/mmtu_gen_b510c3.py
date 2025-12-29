"""MMTU (Massive Multi-Task Table Understanding) Benchmark Configuration.

MMTU is a comprehensive benchmark featuring approximately 28,000 questions
across 25 real-world table tasks, evaluating models' capabilities in
understanding, reasoning about, and manipulating tabular data.

Reference: https://github.com/MMTU-Benchmark/MMTU
Paper: NeurIPS 2025 Benchmark Track
"""

from opencompass.datasets import MMTUDataset, MMTUEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# Reader configuration - defines which columns to use
mmtu_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answer',
)

# Inference configuration - defines the prompt template
mmtu_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Evaluation configuration
mmtu_eval_cfg = dict(
    evaluator=dict(type=MMTUEvaluator),
)

# Main dataset configuration - all tasks
mmtu_datasets = [
    dict(
        type=MMTUDataset,
        abbr='mmtu',
        path='MMTU-benchmark/MMTU',
        reader_cfg=mmtu_reader_cfg,
        infer_cfg=mmtu_infer_cfg,
        eval_cfg=mmtu_eval_cfg,
    ),
]

# Alternative configurations for specific task subsets

# NL-2-SQL focused evaluation
mmtu_nl2sql_datasets = [
    dict(
        type=MMTUDataset,
        abbr='mmtu-nl2sql',
        path='MMTU-benchmark/MMTU',
        task_filter=['NL-2-SQL'],
        reader_cfg=mmtu_reader_cfg,
        infer_cfg=mmtu_infer_cfg,
        eval_cfg=mmtu_eval_cfg,
    ),
]

# Table Question Answering focused evaluation
mmtu_tableqa_datasets = [
    dict(
        type=MMTUDataset,
        abbr='mmtu-tableqa',
        path='MMTU-benchmark/MMTU',
        task_filter=['Table Question Answering'],
        reader_cfg=mmtu_reader_cfg,
        infer_cfg=mmtu_infer_cfg,
        eval_cfg=mmtu_eval_cfg,
    ),
]

# Fact Verification focused evaluation
mmtu_factver_datasets = [
    dict(
        type=MMTUDataset,
        abbr='mmtu-factver',
        path='MMTU-benchmark/MMTU',
        task_filter=['Fact Verification'],
        reader_cfg=mmtu_reader_cfg,
        infer_cfg=mmtu_infer_cfg,
        eval_cfg=mmtu_eval_cfg,
    ),
]

# Data Transformation tasks
mmtu_transform_datasets = [
    dict(
        type=MMTUDataset,
        abbr='mmtu-transform',
        path='MMTU-benchmark/MMTU',
        task_filter=[
            'table-transform-by-relationalization',
            'table-transform-by-output-schema',
            'table-transform-by-output-table',
            'semantic-transform-by-example',
            'program-transform-by-example',
        ],
        reader_cfg=mmtu_reader_cfg,
        infer_cfg=mmtu_infer_cfg,
        eval_cfg=mmtu_eval_cfg,
    ),
]

# Entity/Schema Matching tasks
mmtu_matching_datasets = [
    dict(
        type=MMTUDataset,
        abbr='mmtu-matching',
        path='MMTU-benchmark/MMTU',
        task_filter=[
            'Entity matching',
            'Schema matching',
            'Head value matching',
        ],
        reader_cfg=mmtu_reader_cfg,
        infer_cfg=mmtu_infer_cfg,
        eval_cfg=mmtu_eval_cfg,
    ),
]
