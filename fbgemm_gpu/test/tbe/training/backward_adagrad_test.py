#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import logging
import os
import time
import unittest
from typing import Any

import torch
from fbgemm_gpu.split_embedding_configs import (
    EmbeddingLocation,
    EmbOptimType as OptimType,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import given, settings

from ..common import create_tbe_from_config, load_tbe_configs_from_file  # noqa E402
from .backward_adagrad_common import (
    additional_decorators,
    adjust_mixed_B_st,
    CacheAlgorithm,
    common_settings,
    common_strategy,
    execute_backward_adagrad,
    gpu_unavailable,
    optests,
    PoolingMode,
    SparseType,
    st,
)

# Set up test strategy
test_st: dict[str, Any] = common_strategy.copy()
test_st["D"] = st.integers(min_value=2, max_value=128)
test_st_cpu: dict[str, Any] = test_st.copy()
test_st_cpu["use_cpu"] = st.just(True)
test_st_cpu["row_wise"] = st.just(True)
test_st_cpu["output_dtype"] = st.sampled_from([SparseType.FP32, SparseType.FP16])


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class BackwardAdagradTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(mixed_B=st.booleans(), **test_st)
    @settings(**common_settings)
    def test_backward_adagrad_fp16_pmSUM(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            weights_precision=SparseType.FP16,
            pooling_mode=PoolingMode.SUM,
            compile=False,  # FIXME: make compilation work for fp16
            **kwargs,
        )

    @optests.dontGenerateOpCheckTests("FP8 compute requires custom op support.")
    @unittest.skipIf(*gpu_unavailable)
    @given(mixed_B=st.booleans(), **test_st)
    @settings(**common_settings)
    def test_backward_adagrad_fp8_pmSUM(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        # Skip for use_cpu=True, as FP8 is not supported on CPU.
        # Also disable on AMD for now.
        if kwargs["use_cpu"] or torch.version.hip:
            return
        execute_backward_adagrad(
            weights_precision=SparseType.NFP8,
            pooling_mode=PoolingMode.SUM,
            compile=False,  # FIXME: make compilation work for fp16
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        mixed_B=st.booleans(),
        compile=st.booleans(),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp16_pmMEAN(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            weights_precision=SparseType.FP16,
            pooling_mode=PoolingMode.MEAN,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        compile=st.booleans(),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp16_pmNONE(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        execute_backward_adagrad(
            weights_precision=SparseType.FP16,
            pooling_mode=PoolingMode.NONE,
            mixed_B=False,
            **kwargs,
        )

    @given(
        mixed_B=st.booleans(),
        compile=st.booleans(),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp32_pmSUM(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            weights_precision=SparseType.FP32,
            pooling_mode=PoolingMode.SUM,
            **kwargs,
        )

    @given(
        compile=st.booleans(),
        pooling_mode=st.sampled_from([PoolingMode.SUM, PoolingMode.MEAN]),
        **test_st_cpu,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp32_cpu(  # noqa C901
        self,
        pooling_mode: PoolingMode,
        **kwargs: Any,
    ) -> None:
        """
        Test VBE support for CPU on rowwise adagrad
        """
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            weights_precision=SparseType.FP32,
            pooling_mode=pooling_mode,
            mixed_B=True,
            **kwargs,
        )

    @given(
        compile=st.booleans(),
        **test_st_cpu,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp32_pmNONE_cpu(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        execute_backward_adagrad(
            weights_precision=SparseType.FP32,
            mixed_B=False,
            pooling_mode=PoolingMode.NONE,
            **kwargs,
        )

    @given(
        mixed_B=st.booleans(),
        compile=st.booleans(),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp32_pmMEAN(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        execute_backward_adagrad(
            weights_precision=SparseType.FP32,
            pooling_mode=PoolingMode.MEAN,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        compile=st.booleans(),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp32_pmNONE(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        execute_backward_adagrad(
            weights_precision=SparseType.FP32,
            mixed_B=False,
            pooling_mode=PoolingMode.NONE,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        mixed_B=st.booleans(),
        max_norm=st.floats(min_value=0.01, max_value=1.0),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_fp16_pmSUM_with_max_norm(  # noqa C901
        self,
        **kwargs: Any,
    ) -> None:
        kwargs = adjust_mixed_B_st(kwargs)
        fixed_strategy = {"row_wise": True, "use_cpu": False}
        for key, val in fixed_strategy.items():
            assert key in kwargs
            kwargs[key] = val
        execute_backward_adagrad(
            weights_precision=SparseType.FP16,
            pooling_mode=PoolingMode.SUM,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_backward_adagrad_fp16_pmSUM_D320(self) -> None:
        execute_backward_adagrad(
            T=2,
            # using D=80 since the test harness multiplies D by 4, so 80*4=320
            D=80,
            B=16,
            log_E=4,
            L=4,
            D_gradcheck=1,
            weights_precision=SparseType.FP16,
            stochastic_rounding=False,
            weighted=False,
            row_wise=True,
            mixed=False,
            mixed_B=False,
            use_cache=False,
            cache_algorithm=CacheAlgorithm.LRU,
            pooling_mode=PoolingMode.SUM,
            use_cpu=False,
            output_dtype=SparseType.FP16,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_backward_adagrad_from_config_file(self) -> None:
        """
        Test backward adagrad pass using TBE specs loaded from a JSON file.

        This test reads TBE init parameters from a JSON file, either input_tbe_specs.json by default
        or specified by the TBE_CONFIG_PATH environment variable. It runs backward tests for each
        configuration.

        The config file can be generated using extract_specs_from_log.py:
            python3 ${HOME}/fbsource/fbcode/ai_codesign/nonprod/supadchaya/scripts/fbgemm/extract_specs_from_log.py \
                --specs_log <path_to_specs_log> \
                --planner_log <path_to_planner_log> \
                --output <output_json_path>

        To run this test with a specific config file:
            TBE_CONFIG_PATH=/path/to/config.json \
                buck2 run @//mode/opt fbcode//deeplearning/fbgemm/fbgemm_gpu/test/tbe:backward_adagrad \
                -- -r test_backward_adagrad_from_config_file
        or
            TBE_CONFIG_PATH=/path/to/config.json python3 -m pytest backward_adagrad_test.py \
                -k test_backward_adagrad_from_config_file
        """
        default_config_path = os.path.join(
            os.path.dirname(__file__), "input_tbe_specs.json"
        )
        config_path = os.environ.get("TBE_CONFIG_PATH", default_config_path)
        if not config_path or not os.path.exists(config_path):
            self.skipTest(f"Config file not found: {config_path}")

        batch_size, _, common_config, tbe_configs = load_tbe_configs_from_file(
            config_path
        )

        # Use a smaller batch size for testing to reduce memory usage (otherwise OOM)
        test_batch_size = min(batch_size, 512)
        L = 10 if test_batch_size < batch_size else 2
        max_config = os.environ.get("TBE_MAX_CONFIG", "5")
        max_config = int(max_config) if max_config != "all" else -1
        configs_to_test = tbe_configs[:max_config]

        total_start_time = time.perf_counter()
        end_time = total_start_time

        for i, config in enumerate(configs_to_test):
            with self.subTest(config_index=i):
                start_time = time.perf_counter()
                try:
                    tbe_op = create_tbe_from_config(
                        config, common_config, use_cpu=False
                    )
                    T = len(tbe_op.embedding_specs)
                    max_D = tbe_op.max_D
                    mixed_D = tbe_op.mixed_D
                    execute_backward_adagrad(
                        T=T,
                        D=max_D,
                        B=test_batch_size,
                        log_E=0,
                        L=L,
                        D_gradcheck=1,
                        weights_precision=tbe_op.weights_precision,
                        stochastic_rounding=tbe_op.stochastic_rounding,
                        weighted=False,
                        row_wise=(tbe_op.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD),
                        mixed=mixed_D,
                        mixed_B=False,
                        use_cache=False,
                        cache_algorithm=tbe_op.cache_algorithm,
                        pooling_mode=tbe_op.pooling_mode,
                        use_cpu=False,
                        output_dtype=SparseType.from_int(tbe_op.output_dtype),
                        weight_decay_mode=tbe_op.weight_decay_mode,
                        tbe_op=tbe_op,
                    )
                    end_time = time.perf_counter()
                    logging.info(
                        f"TBE_DEBUG: PASSED for config {i} T={T} max_D={max_D} (mixed: {mixed_D}), the test took {(end_time - start_time) / 60:.2f} mins"
                    )
                except Exception as e:
                    # Log but don't fail - some configs may have unsupported features
                    end_time = time.perf_counter()
                    raise RuntimeError(
                        f"TEST FAILED for config {i} uuid={config.get('tbe_uuid', '')}, the test took {(end_time - start_time) / 60:.2f} mins and failed: {e}"
                    )
        logging.info(
            f"TBE_DEBUG: Total time taken to test {len(configs_to_test)} configs is {(end_time - total_start_time) / 60:.2f} mins"
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_backward_adagrad_with_simple_tbe_op(self) -> None:
        """
        Test execute_backward_adagrad with a pre-created TBE op.

        This test verifies that execute_backward_adagrad works correctly
        with a manually created TBE op passed via the cc parameter.
        """
        T = 1  # Number of tables
        D = 4  # Embedding dimension
        E = 10  # Number of embeddings per table
        B = 1  # Batch size
        L = 1  # Sequence length
        pooling_mode = PoolingMode.SUM
        weights_precision = SparseType.FP32
        output_dtype = SparseType.FP32
        embedding_specs = [
            (E, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA) for _ in range(T)
        ]

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            weights_precision=weights_precision,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            learning_rate=0.05,
            eps=0.2,
            pooling_mode=pooling_mode,
            output_dtype=output_dtype,
        )

        execute_backward_adagrad(
            T=T,
            D=D,
            B=B,
            log_E=1,
            L=L,
            D_gradcheck=1,
            weights_precision=weights_precision,
            stochastic_rounding=False,
            weighted=False,
            row_wise=True,
            mixed=False,
            mixed_B=False,
            use_cache=False,
            cache_algorithm=CacheAlgorithm.LRU,
            pooling_mode=pooling_mode,
            use_cpu=False,
            output_dtype=output_dtype,
            tbe_op=cc,
        )


if __name__ == "__main__":
    unittest.main()
