#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eux

export FBGEMM_TEST_WITH_ROCM=1

python layout_transform_ops_test.py --verbose

python permute_pooled_embedding_modules_test.py --verbose

python sparse_ops_test.py --verbose

python merge_pooled_embeddings_test.py --verbose

python quantize_ops_test.py --verbose

python split_embedding_inference_converter_test.py --verbose

python split_table_batched_embeddings_test.py --verbose