"""MMTU (Massive Multi-Task Table Understanding) Benchmark Dataset.

MMTU is a comprehensive benchmark featuring approximately 28,000 questions
across 25 real-world table tasks. It evaluates models' capabilities in
understanding, reasoning about, and manipulating tabular data at an expert
level.

The benchmark encompasses diverse table operations including:
- Natural language to SQL conversion (NL-2-SQL)
- Table question-answering
- Data transformation and manipulation
- Entity matching and schema matching
- Fact verification
- Error detection and data imputation
- And more complex table reasoning tasks

Reference: https://github.com/MMTU-Benchmark/MMTU
HuggingFace: https://huggingface.co/datasets/MMTU-benchmark/MMTU

Note on Evaluation Differences:
    This implementation uses a simplified unified evaluator with accuracy
    metrics. The official MMTU benchmark uses 27 task-specific evaluators
    with different metrics:
    - Accuracy for: NL2SQL, Table-QA, Entity-Matching, Fact-Verification, etc.
    - F1 score for: Error-Detection, Schema-Matching, Semantic-Join, etc.
    - SQL execution comparison (not string matching) for NL2SQL tasks

    Future enhancements could add task-specific evaluators for more accurate
    alignment with the official benchmark results.
"""

import re
from typing import List, Optional

from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

# MMTU task categories
MMTU_TASK_CATEGORIES = [
    'table-transform-by-relationalization',
    'table-transform-by-output-schema',
    'table-transform-by-output-table',
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
    'NL-2-SQL',
    'Table Question Answering',
    'Fact Verification',
    'Column type annotation',
    'Column property annotation',
    'Cell entity annotation',
]


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    Args:
        text: The answer text to normalize.

    Returns:
        Normalized text string.
    """
    if text is None:
        return ''

    text = str(text).strip().lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove common punctuation at the end
    text = re.sub(r'[.,;:!?]+$', '', text)

    return text


def extract_answer_from_response(response: str) -> str:
    """Extract the answer from a model response.

    Handles various response formats including:
    - "The answer is X"
    - "Answer: X"
    - Direct answer
    - SQL queries
    - JSON/structured outputs

    Args:
        response: The model's response string.

    Returns:
        Extracted answer string.
    """
    if response is None:
        return ''

    response = str(response).strip()

    # Try to extract answer from common patterns
    patterns = [
        r'[Tt]he answer is[:\s]+(.+?)(?:\.|$)',
        r'[Aa]nswer[:\s]+(.+?)(?:\.|$)',
        r'[Rr]esult[:\s]+(.+?)(?:\.|$)',
        r'[Oo]utput[:\s]+(.+?)(?:\.|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no pattern matches, return the full response
    return response


def check_answer_correctness(prediction: str, reference: str,
                             task_type: Optional[str] = None) -> bool:
    """Check if the prediction matches the reference answer.

    Args:
        prediction: The model's predicted answer.
        reference: The ground truth answer.
        task_type: Optional task type for specialized checking.

    Returns:
        Boolean indicating whether the answer is correct.
    """
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    # CRITICAL: Empty predictions should NEVER be considered correct
    # This handles the case where a model produces no output
    if not pred_norm:
        return False

    # If reference is empty but prediction is not, it's incorrect
    if not ref_norm:
        return False

    # Exact match after normalization
    if pred_norm == ref_norm:
        return True

    # For SQL tasks, try more lenient comparison
    if task_type and 'sql' in task_type.lower():
        # Remove extra whitespace and normalize SQL keywords
        pred_sql = ' '.join(pred_norm.upper().split())
        ref_sql = ' '.join(ref_norm.upper().split())
        if pred_sql == ref_sql:
            return True

    # For boolean/yes-no answers
    bool_mappings = {
        'yes': ['yes', 'true', '1', 'correct'],
        'no': ['no', 'false', '0', 'incorrect'],
    }
    for canonical, variants in bool_mappings.items():
        if ref_norm in variants and pred_norm in variants:
            return True

    # Check if prediction contains the reference (but only if reference is non-empty)
    # Note: ref_norm is guaranteed non-empty here due to the check above
    if ref_norm in pred_norm:
        return True

    return False


@ICL_EVALUATORS.register_module()
class MMTUEvaluator(BaseEvaluator):
    """Evaluator for MMTU benchmark.

    Supports evaluation across all 25 task categories with appropriate
    metrics for each task type.
    """

    def score(self, predictions: List, references: List,
              test_set: Optional[Dataset] = None) -> dict:
        """Calculate evaluation metrics.

        Args:
            predictions: List of model predictions.
            references: List of ground truth answers.
            test_set: Optional test dataset with additional metadata.

        Returns:
            Dictionary containing accuracy and per-task metrics.
        """
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different lengths'
            }

        total_correct = 0
        total_count = len(predictions)
        details = []

        # Track per-task statistics
        task_stats = {}

        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            # Get task type if available
            task_type = None
            if test_set is not None and 'task' in test_set.column_names:
                task_type = test_set['task'][idx]
            elif test_set is not None and 'metadata' in test_set.column_names:
                metadata = test_set['metadata'][idx]
                if isinstance(metadata, dict):
                    task_type = metadata.get('task', None)

            # Extract answer from prediction
            extracted_pred = extract_answer_from_response(pred)

            # Get reference answer
            if isinstance(ref, dict):
                ref_answer = ref.get('answer', ref.get('gold', str(ref)))
            else:
                ref_answer = str(ref)

            # Check correctness
            is_correct = check_answer_correctness(
                extracted_pred, ref_answer, task_type)

            if is_correct:
                total_correct += 1

            # Track per-task statistics
            if task_type:
                if task_type not in task_stats:
                    task_stats[task_type] = {'correct': 0, 'total': 0}
                task_stats[task_type]['total'] += 1
                if is_correct:
                    task_stats[task_type]['correct'] += 1

            # Record details
            detail = {
                'prediction': pred,
                'extracted': extracted_pred,
                'reference': ref_answer,
                'correct': is_correct,
            }
            if task_type:
                detail['task'] = task_type
            details.append(detail)

        # Calculate overall accuracy
        accuracy = 100.0 * total_correct / total_count if total_count > 0 else 0.0

        # Build results dictionary
        results = {
            'accuracy': round(accuracy, 2),
            'correct': total_correct,
            'total': total_count,
            'details': details,
        }

        # Add per-task accuracies
        for task, stats in task_stats.items():
            if stats['total'] > 0:
                task_acc = 100.0 * stats['correct'] / stats['total']
                # Create a safe key name for the task
                safe_task_name = task.replace(' ', '_').replace('-', '_').lower()
                results[f'mmtu_{safe_task_name}'] = round(task_acc, 2)

        return results


@LOAD_DATASET.register_module()
class MMTUDataset(BaseDataset):
    """Dataset loader for MMTU benchmark.

    Loads the MMTU dataset from HuggingFace and prepares it for evaluation.
    The dataset contains ~28,000 questions across 25 table task categories.

    Example usage in config:
        dict(
            type=MMTUDataset,
            path='MMTU-benchmark/MMTU',
            task_filter=['NL-2-SQL', 'Table Question Answering'],
        )
    """

    @staticmethod
    def load(path: str = 'MMTU-benchmark/MMTU',
             split: str = 'train',
             task_filter: Optional[List[str]] = None,
             **kwargs) -> Dataset:
        """Load the MMTU dataset.

        Args:
            path: HuggingFace dataset path or local path.
            split: Dataset split to load (default: 'train').
            task_filter: Optional list of task categories to include.
                If None, all tasks are included.
            **kwargs: Additional arguments passed to load_dataset.

        Returns:
            HuggingFace Dataset object with processed data.
        """
        # Load from HuggingFace
        dataset = load_dataset(path, split=split, trust_remote_code=True,
                               **kwargs)

        # Process each item
        def process_item(item):
            """Process a single dataset item."""
            processed = {}

            # Handle prompt/question field
            if 'prompt' in item:
                processed['prompt'] = item['prompt']
            elif 'question' in item:
                processed['prompt'] = item['question']
            else:
                # Fallback: use first string field as prompt
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 10:
                        processed['prompt'] = value
                        break

            # Handle metadata field
            if 'metadata' in item:
                metadata = item['metadata']
                if isinstance(metadata, dict):
                    processed['task'] = metadata.get('task', 'unknown')
                    processed['metadata'] = metadata
                else:
                    processed['metadata'] = str(metadata)
                    processed['task'] = 'unknown'
            else:
                processed['task'] = 'unknown'
                processed['metadata'] = {}

            # Handle answer/gold field
            if 'answer' in item:
                processed['answer'] = item['answer']
            elif 'gold' in item:
                processed['answer'] = item['gold']
            elif 'expected' in item:
                processed['answer'] = item['expected']
            else:
                processed['answer'] = ''

            return processed

        # Apply processing
        dataset = dataset.map(process_item)

        # Filter by task if specified
        if task_filter is not None:
            task_filter_lower = [t.lower() for t in task_filter]
            dataset = dataset.filter(
                lambda x: x.get('task', '').lower() in task_filter_lower
            )

        return dataset
