"""MMTU (Massive Multi-Task Table Understanding) Benchmark Dataset.

MMTU is a large-scale benchmark with around 28K questions across 25 real-world
table tasks, designed to comprehensively evaluate models' ability to understand,
reason, and manipulate real tables at the expert-level.

Paper: https://arxiv.org/abs/2505.11125
GitHub: https://github.com/MMTU-Benchmark/MMTU
HuggingFace: https://huggingface.co/datasets/MMTU-benchmark/MMTU
"""

import json
import os
from typing import List, Optional

from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

# MMTU task categories
MMTU_TASKS = [
    'NL-2-SQL',
    'Table Question Answering',
    'Fact Verification',
    'Column type annotation',
    'Column property annotation',
    'Cell entity annotation',
    'Entity matching',
    'Schema matching',
    'Head value matching',
    'data-imputation',
    'error-detection',
    'list-to-table',
    'semantic-join',
    'equi-join-detect',
    'program-transform-by-example',
    'formula-by-context',
    'semantic-transform-by-example',
    'arithmetic-relationship',
    'functional-relationship',
    'string-relationship',
    'Needle-in-a-haystack-table',
    'Needle-in-a-haystack-index',
    'table-transform-by-relationalization',
    'table-transform-by-output-schema',
    'table-transform-by-output-table',
]


def parse_metadata(metadata_str: str) -> dict:
    """Parse metadata string to dictionary.

    Args:
        metadata_str: JSON string containing metadata

    Returns:
        Parsed metadata dictionary
    """
    if isinstance(metadata_str, dict):
        return metadata_str
    try:
        return json.loads(metadata_str)
    except (json.JSONDecodeError, TypeError):
        return {}


@LOAD_DATASET.register_module()
class MMTUDataset(BaseDataset):
    """MMTU Dataset loader.

    Loads the MMTU benchmark dataset from HuggingFace or local path.
    """

    @staticmethod
    def load(path: str,
             task: Optional[str] = None,
             split: str = 'test',
             **kwargs) -> Dataset:
        """Load MMTU dataset.

        Args:
            path: Path to dataset or HuggingFace dataset ID
            task: Optional task filter (e.g., 'NL-2-SQL', 'Table Question Answering')
            split: Dataset split to load
            **kwargs: Additional arguments passed to load_dataset

        Returns:
            Dataset object with processed MMTU data
        """
        # Try loading from HuggingFace first
        try:
            dataset = load_dataset(path, split=split, **kwargs)
        except Exception:
            # Fall back to local path
            path = get_data_path(path, local_mode=True)
            if os.path.isdir(path):
                # Load from JSONL file
                jsonl_path = os.path.join(path, 'mmtu.jsonl')
                if os.path.exists(jsonl_path):
                    dataset = load_dataset('json',
                                           data_files=jsonl_path,
                                           split='train')
                else:
                    # Try loading all jsonl files in directory
                    jsonl_files = [
                        os.path.join(path, f) for f in os.listdir(path)
                        if f.endswith('.jsonl')
                    ]
                    if jsonl_files:
                        dataset = load_dataset('json',
                                               data_files=jsonl_files,
                                               split='train')
                    else:
                        raise FileNotFoundError(
                            f'No JSONL files found in {path}')
            else:
                dataset = load_dataset('json', data_files=path, split='train')

        # Process dataset
        processed_data = []
        for item in dataset:
            # Parse metadata if it's a string
            metadata = parse_metadata(item.get('metadata', {}))

            # Get task from metadata
            item_task = metadata.get('task', '')

            # Filter by task if specified
            if task and item_task != task:
                continue

            processed_item = {
                'prompt': item.get('prompt', ''),
                'task': item_task,
                'metadata': json.dumps(metadata)
                if isinstance(metadata, dict) else str(metadata),
            }

            # Add answer/ground_truth if available
            if 'answer' in item:
                processed_item['answer'] = item['answer']
            elif 'ground_truth' in item:
                processed_item['answer'] = item['ground_truth']
            elif 'expected_output' in metadata:
                processed_item['answer'] = metadata['expected_output']
            else:
                processed_item['answer'] = ''

            processed_data.append(processed_item)

        return Dataset.from_list(processed_data)


@ICL_EVALUATORS.register_module()
class MMTUEvaluator(BaseEvaluator):
    """MMTU Evaluator.

    Evaluates model predictions against ground truth answers using
    exact match and normalized string comparison.
    """

    def __init__(self, metric: str = 'accuracy', **kwargs):
        """Initialize MMTU evaluator.

        Args:
            metric: Evaluation metric ('accuracy' or 'f1')
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.metric = metric

    def score(self, predictions: List[str], references: List[str],
              **kwargs) -> dict:
        """Score predictions against references.

        Args:
            predictions: Model predictions
            references: Ground truth answers
            **kwargs: Additional arguments (may include test_set)

        Returns:
            Dictionary containing score and details
        """
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different length'
            }

        correct = 0
        details = []

        for pred, ref in zip(predictions, references):
            # Normalize strings for comparison
            pred_normalized = self._normalize(pred)
            ref_normalized = self._normalize(ref)

            # Check for exact match
            is_correct = pred_normalized == ref_normalized

            # Also check if prediction contains the reference
            # (for SQL and code outputs)
            if not is_correct and ref_normalized:
                is_correct = ref_normalized in pred_normalized

            if is_correct:
                correct += 1

            details.append({
                'pred': pred,
                'answer': ref,
                'correct': is_correct,
            })

        accuracy = (correct / len(predictions) * 100) if predictions else 0

        return {
            'accuracy': accuracy,
            'score': accuracy,  # Alias for compatibility
            'correct': correct,
            'total': len(predictions),
            'details': details,
        }

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase
        text = text.lower().strip()

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove common formatting differences
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')

        return text


@ICL_EVALUATORS.register_module()
class MMTUTaskEvaluator(BaseEvaluator):
    """Task-specific MMTU Evaluator.

    Provides task-aware evaluation with per-task metrics.
    """

    def __init__(self, **kwargs):
        """Initialize task-specific evaluator."""
        super().__init__(**kwargs)

    def score(self, predictions: List[str], references: List[str],
              test_set=None, **kwargs) -> dict:
        """Score predictions with task breakdown.

        Args:
            predictions: Model predictions
            references: Ground truth answers
            test_set: Test dataset with task information
            **kwargs: Additional arguments

        Returns:
            Dictionary with overall and per-task metrics
        """
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different length'
            }

        # Track results overall and per task
        results = {'correct': 0, 'total': 0}
        task_results = {}
        details = []

        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            # Get task type if available
            task = 'unknown'
            if test_set is not None:
                try:
                    metadata = test_set['metadata'][idx]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    task = metadata.get('task', 'unknown')
                except (KeyError, json.JSONDecodeError, IndexError):
                    pass

            # Initialize task tracking
            if task not in task_results:
                task_results[task] = {'correct': 0, 'total': 0}

            # Evaluate
            is_correct = self._evaluate(pred, ref, task)

            results['total'] += 1
            task_results[task]['total'] += 1

            if is_correct:
                results['correct'] += 1
                task_results[task]['correct'] += 1

            details.append({
                'pred': pred,
                'answer': ref,
                'task': task,
                'correct': is_correct,
            })

        # Calculate overall accuracy
        accuracy = (results['correct'] / results['total'] *
                    100) if results['total'] > 0 else 0

        result = {
            'accuracy': accuracy,
            'score': accuracy,
            'correct': results['correct'],
            'total': results['total'],
            'details': details,
        }

        # Add per-task accuracies
        for task, task_res in task_results.items():
            if task_res['total'] > 0:
                task_acc = task_res['correct'] / task_res['total'] * 100
                safe_task_name = task.replace(' ', '_').replace('-', '_')
                result[f'accuracy_{safe_task_name}'] = task_acc

        return result

    def _evaluate(self, pred: str, ref: str, task: str) -> bool:
        """Evaluate a single prediction.

        Args:
            pred: Model prediction
            ref: Ground truth answer
            task: Task type

        Returns:
            True if prediction is correct
        """
        # Normalize both strings
        pred_norm = self._normalize(pred)
        ref_norm = self._normalize(ref)

        # Exact match
        if pred_norm == ref_norm:
            return True

        # For SQL tasks, try semantic comparison
        if 'sql' in task.lower():
            return self._compare_sql(pred, ref)

        # For binary tasks (e.g., fact verification)
        if task.lower() in ['fact verification', 'error-detection']:
            return self._compare_binary(pred, ref)

        # Check if reference is contained in prediction
        if ref_norm and ref_norm in pred_norm:
            return True

        return False

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower().strip()
        text = ' '.join(text.split())
        return text

    def _compare_sql(self, pred: str, ref: str) -> bool:
        """Compare SQL queries."""
        # Basic SQL normalization
        def normalize_sql(sql):
            sql = sql.lower()
            # Remove extra whitespace
            sql = ' '.join(sql.split())
            # Remove trailing semicolon
            sql = sql.rstrip(';')
            return sql

        return normalize_sql(pred) == normalize_sql(ref)

    def _compare_binary(self, pred: str, ref: str) -> bool:
        """Compare binary answers (yes/no, true/false)."""
        pred_lower = pred.lower().strip()
        ref_lower = ref.lower().strip()

        # Map various forms to canonical values
        positive = {'yes', 'true', '1', 'correct', 'valid'}
        negative = {'no', 'false', '0', 'incorrect', 'invalid'}

        pred_is_positive = any(p in pred_lower for p in positive)
        pred_is_negative = any(n in pred_lower for n in negative)
        ref_is_positive = any(p in ref_lower for p in positive)
        ref_is_negative = any(n in ref_lower for n in negative)

        if pred_is_positive and not pred_is_negative:
            return ref_is_positive and not ref_is_negative
        if pred_is_negative and not pred_is_positive:
            return ref_is_negative and not ref_is_positive

        return False
