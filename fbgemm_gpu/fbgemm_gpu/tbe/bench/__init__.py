#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch

from .bench_config import (  # noqa F401
    TBEBenchmarkingConfig,
    TBEBenchmarkingConfigLoader,
    TBEBenchmarkingHelperText,
)
from .bench_runs import (  # noqa F401
    bench_warmup,
    benchmark_cpu_requests,
    benchmark_cpu_requests_mp,
    benchmark_pipelined_requests,
    benchmark_requests,
    benchmark_requests_refer,
    benchmark_vbe,
)
from .benchmark_click_interface import TbeBenchClickInterface  # noqa F401
from .embedding_ops_common_config import EmbeddingOpsCommonConfigLoader  # noqa F401
from .eval_compression import (  # noqa F401
    benchmark_eval_compression,
    EvalCompressionBenchmarkOutput,
)
from .reporter import BenchmarkReporter  # noqa F401
from .tbe_data_config import TBEDataConfig  # noqa F401
from .tbe_data_config_loader import (  # noqa F401
    TBEDataConfigHelperText,
    TBEDataConfigLoader,
)
from .tbe_data_config_param_models import (  # noqa F401
    BatchParams,
    IndicesParams,
    PoolingParams,
)
from .utils import fill_random_scale_bias  # noqa F401

try:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/src/tbe/eeg:indices_estimator"
    )
except Exception:
    pass

#: The max number of heavy heavy hitters, as defined in
#: fbgemm_gpu/src/tbe/eeg/indices_estimator.h
EEG_MAX_HEAVY_HITTERS: int = 20
