#!/bin/bash

set -e

echo "======================================================"
echo "STEP 1: SYNTAX VALIDATED TEACHER GENERATION"
echo "======================================================"

python scripts/generate_syntax_validated_mbpp.py

echo ""
echo "======================================================"
echo "STEP 2: FINDING LATEST SYNTAX PARQUET"
echo "======================================================"

LATEST_PARQUET=$(ls -t data/teacher_outputs/*.parquet | head -n 1)

echo "Latest parquet:"
echo "$LATEST_PARQUET"

echo ""
echo "======================================================"
echo "STEP 3: EXECUTION VALIDATION"
echo "======================================================"

python scripts/generate_execution_validated_mbpp.py \
  --input_parquet "$LATEST_PARQUET"

echo ""
echo "======================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "======================================================"