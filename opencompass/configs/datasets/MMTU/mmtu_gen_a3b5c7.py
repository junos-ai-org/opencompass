"""MMTU (Massive Multi-Task Table Understanding) Benchmark Configuration.

MMTU is a large-scale benchmark with around 28K questions across 25 real-world
table tasks, designed to evaluate models' ability to understand, reason, and
manipulate tables at the expert-level.

Paper: https://arxiv.org/abs/2505.11125
GitHub: https://github.com/MMTU-Benchmark/MMTU
"""

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

from opencompass.datasets import MMTUDataset, MMTUEvaluator, MMTUTaskEvaluator

# MMTU task categories with descriptive names
MMTU_TASKS = {
    'all': {
        'name': 'mmtu',
        'task': None,  # Load all tasks
    },
    'nl2sql': {
        'name': 'mmtu_nl2sql',
        'task': 'NL-2-SQL',
    },
    'tableqa': {
        'name': 'mmtu_tableqa',
        'task': 'Table Question Answering',
    },
    'fact_verification': {
        'name': 'mmtu_fact_verification',
        'task': 'Fact Verification',
    },
    'column_type_annotation': {
        'name': 'mmtu_column_type_annotation',
        'task': 'Column type annotation',
    },
    'column_property_annotation': {
        'name': 'mmtu_column_property_annotation',
        'task': 'Column property annotation',
    },
    'cell_entity_annotation': {
        'name': 'mmtu_cell_entity_annotation',
        'task': 'Cell entity annotation',
    },
    'entity_matching': {
        'name': 'mmtu_entity_matching',
        'task': 'Entity matching',
    },
    'schema_matching': {
        'name': 'mmtu_schema_matching',
        'task': 'Schema matching',
    },
    'head_value_matching': {
        'name': 'mmtu_head_value_matching',
        'task': 'Head value matching',
    },
    'data_imputation': {
        'name': 'mmtu_data_imputation',
        'task': 'data-imputation',
    },
    'error_detection': {
        'name': 'mmtu_error_detection',
        'task': 'error-detection',
    },
    'list_to_table': {
        'name': 'mmtu_list_to_table',
        'task': 'list-to-table',
    },
    'semantic_join': {
        'name': 'mmtu_semantic_join',
        'task': 'semantic-join',
    },
    'equi_join_detect': {
        'name': 'mmtu_equi_join_detect',
        'task': 'equi-join-detect',
    },
    'program_transform': {
        'name': 'mmtu_program_transform',
        'task': 'program-transform-by-example',
    },
    'formula_context': {
        'name': 'mmtu_formula_context',
        'task': 'formula-by-context',
    },
    'semantic_transform': {
        'name': 'mmtu_semantic_transform',
        'task': 'semantic-transform-by-example',
    },
    'arithmetic_relationship': {
        'name': 'mmtu_arithmetic_relationship',
        'task': 'arithmetic-relationship',
    },
    'functional_relationship': {
        'name': 'mmtu_functional_relationship',
        'task': 'functional-relationship',
    },
    'string_relationship': {
        'name': 'mmtu_string_relationship',
        'task': 'string-relationship',
    },
    'needle_haystack_table': {
        'name': 'mmtu_needle_haystack_table',
        'task': 'Needle-in-a-haystack-table',
    },
    'needle_haystack_index': {
        'name': 'mmtu_needle_haystack_index',
        'task': 'Needle-in-a-haystack-index',
    },
    'transform_relationalization': {
        'name': 'mmtu_transform_relationalization',
        'task': 'table-transform-by-relationalization',
    },
    'transform_output_schema': {
        'name': 'mmtu_transform_output_schema',
        'task': 'table-transform-by-output-schema',
    },
    'transform_output_table': {
        'name': 'mmtu_transform_output_table',
        'task': 'table-transform-by-output-table',
    },
}

# Reader configuration
mmtu_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answer',
)

# Inference configuration
mmtu_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{prompt}'),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        max_out_len=2048,
    ),
)

# Evaluation configuration (default with task breakdown)
mmtu_eval_cfg = dict(
    evaluator=dict(type=MMTUTaskEvaluator),
    pred_role='BOT',
)

# Simple evaluation configuration (just accuracy)
mmtu_eval_cfg_simple = dict(
    evaluator=dict(type=MMTUEvaluator),
    pred_role='BOT',
)

# Build dataset configurations
mmtu_datasets = []

# Main MMTU dataset (all tasks)
mmtu_datasets.append(
    dict(
        type=MMTUDataset,
        abbr='mmtu',
        path='MMTU-benchmark/MMTU',
        task=None,  # All tasks
        reader_cfg=mmtu_reader_cfg,
        infer_cfg=mmtu_infer_cfg,
        eval_cfg=mmtu_eval_cfg,
    ))

# Individual task datasets
for task_key, task_info in MMTU_TASKS.items():
    if task_key == 'all':
        continue  # Already added above

    mmtu_datasets.append(
        dict(
            type=MMTUDataset,
            abbr=task_info['name'],
            path='MMTU-benchmark/MMTU',
            task=task_info['task'],
            reader_cfg=mmtu_reader_cfg,
            infer_cfg=mmtu_infer_cfg,
            eval_cfg=mmtu_eval_cfg_simple,
        ))

# Summary groups for aggregated metrics
mmtu_summary_groups = [
    {
        'name': 'mmtu',
        'subsets': [task_info['name'] for task_info in MMTU_TASKS.values()],
    },
    {
        'name': 'mmtu_annotation',
        'subsets': [
            'mmtu_column_type_annotation',
            'mmtu_column_property_annotation',
            'mmtu_cell_entity_annotation',
        ],
    },
    {
        'name': 'mmtu_matching',
        'subsets': [
            'mmtu_entity_matching',
            'mmtu_schema_matching',
            'mmtu_head_value_matching',
        ],
    },
    {
        'name': 'mmtu_transform',
        'subsets': [
            'mmtu_transform_relationalization',
            'mmtu_transform_output_schema',
            'mmtu_transform_output_table',
            'mmtu_program_transform',
            'mmtu_semantic_transform',
        ],
    },
    {
        'name': 'mmtu_reasoning',
        'subsets': [
            'mmtu_nl2sql',
            'mmtu_tableqa',
            'mmtu_fact_verification',
            'mmtu_formula_context',
        ],
    },
    {
        'name': 'mmtu_relationship',
        'subsets': [
            'mmtu_arithmetic_relationship',
            'mmtu_functional_relationship',
            'mmtu_string_relationship',
        ],
    },
]
