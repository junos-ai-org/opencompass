#!/bin/bash
# =============================================================================
# LLaDA Evaluation Script for OpenCompass
# =============================================================================
#
# This script runs various evaluations for LLaDA (Large Language Diffusion
# with mAsking) models using OpenCompass.
#
# LLaDA is a diffusion-based language model that generates text through
# iterative denoising of masked tokens.
#
# Reference: https://arxiv.org/abs/2502.09992
# Official repo: https://github.com/ML-GSAI/LLaDA
#
# =============================================================================
# Decoding Styles:
# =============================================================================
# 1. Confidence-based (diff_confidence_eos_eot_inf=True)
#    - Best for: Math, reasoning tasks (GSM8K, MATH)
#    - Sets confidence of EOS/EOT tokens to -inf during remasking
#
# 2. Logits-based (diff_logits_eos_inf=True)
#    - Best for: Code generation (HumanEval, MBPP)
#    - Sets logits of EOS token to -inf
#
# 3. Basic (both flags False)
#    - Default decoding without special EOS handling
#    - Good for: General tasks (MMLU, HellaSwag)
#
# =============================================================================
# Block Size Impact:
# =============================================================================
# - Larger block_size (e.g., 512): More parallel, faster but may be less accurate
# - Smaller block_size (e.g., 64): More autoregressive, slower but may be better
# - block_size must divide gen_length evenly
#
# =============================================================================

set -e  # Exit on error

# Change to OpenCompass directory
cd "$(dirname "$0")/.."

echo "=============================================="
echo "LLaDA Evaluation Suite"
echo "=============================================="

# =============================================================================
# GSM8K Evaluations (Confidence-based decoding)
# =============================================================================
# Different block sizes to compare parallel vs autoregressive trade-offs

echo ""
echo "[1/5] GSM8K - Confidence decoding (block_size=512, fully parallel)"
python run.py examples/eval_llada_gsm8k_len512_block512_confidence.py \
    -w outputs/llada_gsm8k_len512_block512_confidence

echo ""
echo "[2/5] GSM8K - Confidence decoding (block_size=256, semi-autoregressive)"
python run.py examples/eval_llada_gsm8k_len512_block256_confidence.py \
    -w outputs/llada_gsm8k_len512_block256_confidence

echo ""
echo "[3/5] GSM8K - Confidence decoding (block_size=64, more autoregressive)"
python run.py examples/eval_llada_gsm8k_len512_block64_confidence.py \
    -w outputs/llada_gsm8k_len512_block64_confidence

# =============================================================================
# HumanEval Evaluation (Logits-based decoding)
# =============================================================================

echo ""
echo "[4/5] HumanEval - Logits decoding (for code generation)"
python run.py examples/eval_llada_humaneval_len512_block512_logits.py \
    -w outputs/llada_humaneval_len512_block512_logits

# =============================================================================
# MMLU Evaluation (Basic decoding)
# =============================================================================

echo ""
echo "[5/5] MMLU - Basic decoding"
python run.py examples/eval_llada_mmlu_len256_block256_basic.py \
    -w outputs/llada_mmlu_len256_block256_basic

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "Results saved in outputs/ directory"
echo "=============================================="
