# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Worker for nbit_forward_threading_test: builds a deterministic CPU int-nbit TBE op
# and writes its forward output to the path given as argv[1]. The driver runs this
# under different FBGEMM_TBE_MAX_NUM_THREADS / FBGEMM_TBE_MIN_TABLES_PER_THREAD env values (read once,
# at the first kernel call, hence a separate process per setting) and checks the
# outputs are bitwise identical -- i.e. table-threading does not change the result.
import sys

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)


def main() -> None:
    out_path = sys.argv[1]
    # T=40 > the default threading onset (2*G = 32 at G=16), so even the
    # default-granularity arm (FBGEMM_TBE_MAX_NUM_THREADS=2, no FBGEMM_TBE_MIN_TABLES_PER_THREAD)
    # genuinely spawns threads rather than falling back to the serial path.
    T, E, D, B, L = 40, 1000, 16, 8, 6

    # Deterministic weights: same seed + same torch build => identical across the
    # worker processes the driver spawns, so the only variable is the thread count.
    torch.manual_seed(0)
    cc = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[("", E, D, SparseType.INT8, EmbeddingLocation.HOST)] * T,
        pooling_mode=PoolingMode.SUM,
        device="cpu",
        output_dtype=SparseType.FP16,
    )
    cc.fill_random_weights()

    # Deterministic indices/offsets (no RNG): T*B bags, each pooling L indices.
    indices = (torch.arange(T * B * L) % E).to(torch.int32)
    offsets = (torch.arange(T * B + 1) * L).to(torch.int32)

    out = cc(indices, offsets)
    torch.save(out.cpu(), out_path)


if __name__ == "__main__":
    main()
