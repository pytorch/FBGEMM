# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]


import unittest

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BackendType,
    PoolingMode,
)

from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from hypothesis import assume, given, settings, Verbosity

from .. import common  # noqa E402
from ..common import gpu_unavailable, running_in_oss
from .training_common import (
    default_strategies,
    MAX_EXAMPLES,
    SSDSplitTableBatchedEmbeddingsTestCommon,
)


@unittest.skipIf(*running_in_oss)
@unittest.skipIf(*gpu_unavailable)
class SSDSplitTBERowwiseAdagradTest(SSDSplitTableBatchedEmbeddingsTestCommon):
    @given(
        **default_strategies,
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_backward_adagrad(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        backend_type: BackendType,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
        ) = self.generate_ssd_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            backend_type=backend_type,
        )

        Es = [emb.embedding_specs[t][0] for t in range(T)]
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
        )

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare optimizer states
        split_optimizer_states = self.split_optimizer_states_(emb)
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_optimizer_state = emb_ref[f].weight.grad.float().to_dense().pow(2)
            torch.testing.assert_close(
                split_optimizer_states[t][0].float(),
                ref_optimizer_state.mean(dim=1),
                atol=tolerance,
                rtol=tolerance,
            )

        # Compare weights
        emb.flush()

        emb_test = emb.debug_split_embedding_weights()
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            emb_r = emb_ref[f]
            (m1,) = split_optimizer_states[t]
            new_ref_weight = torch.addcdiv(
                emb_r.weight.float(),
                value=-lr,
                tensor1=emb_r.weight.grad.float().to_dense(),
                tensor2=m1.float().sqrt_().add_(eps).view(Es[t], 1),
            )

            if weights_precision == SparseType.FP16:
                # Round the reference weight the same way that TBE does
                new_ref_weight = new_ref_weight.half().float()

            torch.testing.assert_close(
                emb_test[t].float().cuda(),
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        bulk_init_chunk_size=st.sampled_from([0, 204800]),
        lazy_bulk_init_enabled=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_emb_state_dict_adagrad(
        self, bulk_init_chunk_size: int, lazy_bulk_init_enabled: bool
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        T = 4
        B = 10
        D = 128
        L = 10
        log_E = 4
        weights_precision = SparseType.FP32
        output_dtype = SparseType.FP32
        pooling_mode = PoolingMode.SUM

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
        ) = self.generate_ssd_tbes(
            T,
            D,
            B,
            log_E,
            L,
            False,  # weighted
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            cache_set_scale=0.2,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=True,
            bulk_init_chunk_size=bulk_init_chunk_size,
            lazy_bulk_init_enabled=lazy_bulk_init_enabled,
        )

        Es = [emb.embedding_specs[t][0] for t in range(T)]
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=True,
            mixed_B=True,
        )

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            False,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        split_optimizer_states = self.split_optimizer_states_(emb)

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict, _, _, _ = emb.split_embedding_weights(no_snapshot=False)
        for feature_index, table_index in self.get_physical_table_arg_indices_(
            emb.feature_table_map
        ):
            (m1,) = split_optimizer_states[table_index]
            emb_r = emb_ref[feature_index]
            self.assertLess(table_index, len(emb_state_dict))
            new_ref_weight = torch.addcdiv(
                emb_r.weight.float(),
                value=-lr,
                tensor1=emb_r.weight.grad.float().to_dense(),  # pyre-ignore[16]
                tensor2=m1.float().sqrt_().add_(eps).view(Es[table_index], 1),
            ).cpu()

            torch.testing.assert_close(
                # pyre-fixme[16]: Undefined attribute: Item `torch._tensor.Tensor` of `typing.Uni...
                emb_state_dict[table_index].full_tensor().float(),
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_strategies,
        num_buckets=st.integers(min_value=10, max_value=15),
        enable_optimizer_offloading=st.booleans(),
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_emb_state_dict_adagrad(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        enable_optimizer_offloading: bool,
        backend_type: BackendType,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_type=backend_type,
            mixed=True,
        )

        # Generate inputs
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
        )

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        split_optimizer_states = []

        # Compare emb state dict with expected values from nn.EmbeddingBag
        (
            emb_state_dict_list,
            bucket_asc_ids_list,
            num_active_id_per_bucket_list,
            metadata_list,
        ) = emb.split_embedding_weights(no_snapshot=False, should_flush=True)

        for s in emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        ):
            split_optimizer_states.append(s)

        # Compare optimizer states
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`
            ref_optimizer_state = emb_ref[f].weight.grad.float().to_dense().pow(2)

            # pyre-fixme[16]: Undefined attribute: `Optional` has no attribute `__getitem__`.
            ref_opt_mean = ref_optimizer_state[bucket_asc_ids_list[t].view(-1)].mean(
                dim=1
            )
            torch.testing.assert_close(
                # NOTE: The [0] index is a hack since the test is fixed to use
                # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
                # be upgraded in the future to support multiple optimizers
                split_optimizer_states[t][0].float(),
                ref_opt_mean.cpu(),
                atol=tolerance,
                rtol=tolerance,
            )

        for feature_index, table_index in self.get_physical_table_arg_indices_(
            emb.feature_table_map
        ):
            """
            validate bucket_asc_ids_list and num_active_id_per_bucket_list
            """
            bucket_asc_id = bucket_asc_ids_list[table_index]
            num_active_id_per_bucket = num_active_id_per_bucket_list[table_index]

            bucket_id_start = bucket_offsets[table_index][0]
            bucket_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_active_id_per_bucket.view(-1)
            )
            for bucket_idx, id_count in enumerate(num_active_id_per_bucket):
                bucket_id = bucket_idx + bucket_id_start
                active_id_cnt = 0
                for idx in range(
                    bucket_id_offsets[bucket_idx],
                    bucket_id_offsets[bucket_idx + 1],
                ):
                    # for chunk-based hashing
                    self.assertEqual(
                        bucket_id, bucket_asc_id[idx] // bucket_sizes[table_index]
                    )
                    active_id_cnt += 1
                self.assertEqual(active_id_cnt, id_count)

            """
            validate embeddings
            """
            num_ids = len(bucket_asc_ids_list[table_index])
            emb_r_w = emb_ref[feature_index].weight[
                bucket_asc_ids_list[table_index].view(-1)
            ]
            emb_r_w_g = (
                emb_ref[feature_index]
                .weight.grad.float()
                .to_dense()[bucket_asc_ids_list[table_index].view(-1)]
            )
            self.assertLess(table_index, len(emb_state_dict_list))
            assert len(split_optimizer_states[table_index][0]) == num_ids
            (m1,) = split_optimizer_states[table_index]
            new_ref_weight = torch.addcdiv(
                emb_r_w.float(),
                value=-lr,
                tensor1=emb_r_w_g,
                tensor2=m1.float().sqrt_().add_(eps).view(num_ids, 1).cuda(),
            ).cpu()

            emb_w = (
                emb_state_dict_list[table_index]
                .narrow(0, 0, bucket_asc_ids_list[table_index].size(0))
                .float()
            )
            torch.testing.assert_close(
                emb_w,
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

            self.assertTrue(len(metadata_list[table_index].size()) == 2)

    @given(
        **default_strategies,
        num_buckets=st.integers(min_value=10, max_value=15),
        enable_optimizer_offloading=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_opt_state_w_offloading_adagrad(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        enable_optimizer_offloading: bool,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # TODO: check split_optimizer_states when optimizer offloading is ready
        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
        )

        # Generate inputs
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
        )

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )
        split_optimizer_states = emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False
        )
        table_input_id_range = []
        for t, row in enumerate(Es):
            bucket_id_start = bucket_offsets[t][0]
            bucket_id_end = bucket_offsets[t][1]
            bucket_size = bucket_sizes[t]
            table_input_id_range.append(
                (
                    min(bucket_id_start * bucket_size, row),
                    min(bucket_id_end * bucket_size, row),
                )
            )
            # since we use ref_emb in dense format, the rows start from id 0
            self.assertEqual(table_input_id_range[-1][0], 0)

        # Compare optimizer states
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_emb = emb_ref[f].weight.grad.float().to_dense().pow(2).cpu()
            ref_optimizer_state = ref_emb.mean(dim=1)[
                table_input_id_range[t][0] : min(
                    table_input_id_range[t][1], emb_ref[f].weight.size(0)
                )
            ]
            # pyre-fixme[16]: Undefined attribute: `Optional` has no attribute `__getitem__`.
            ref_kv_opt = ref_optimizer_state[bucket_asc_ids_list[t]].view(-1)
            torch.testing.assert_close(
                # NOTE: The [0] index is a hack since the test is fixed to use
                # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
                # be upgraded in the future to support multiple optimizers
                split_optimizer_states[t][0].float(),
                ref_kv_opt,
                atol=tolerance,
                rtol=tolerance,
            )

        for feature_index, table_index in self.get_physical_table_arg_indices_(
            emb.feature_table_map
        ):
            """
            validate bucket_asc_ids_list and num_active_id_per_bucket_list
            """
            bucket_asc_id = bucket_asc_ids_list[table_index]
            num_active_id_per_bucket = num_active_id_per_bucket_list[table_index]

            bucket_id_start = bucket_offsets[table_index][0]
            bucket_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_active_id_per_bucket.view(-1)
            )
            for bucket_idx, id_count in enumerate(num_active_id_per_bucket):
                bucket_id = bucket_idx + bucket_id_start
                active_id_cnt = 0
                for idx in range(
                    bucket_id_offsets[bucket_idx],
                    bucket_id_offsets[bucket_idx + 1],
                ):
                    # for chunk-based hashing
                    self.assertEqual(
                        bucket_id, bucket_asc_id[idx] // bucket_sizes[table_index]
                    )
                    active_id_cnt += 1
                self.assertEqual(active_id_cnt, id_count)

            """
            validate embeddings
            """
            num_ids = len(bucket_asc_ids_list[table_index])
            emb_r_w = emb_ref[feature_index].weight[
                bucket_asc_ids_list[table_index].view(-1)
            ]
            emb_r_w_g = (
                emb_ref[feature_index]
                .weight.grad.float()
                .to_dense()[bucket_asc_ids_list[table_index].view(-1)]
            )
            self.assertLess(table_index, len(emb_state_dict_list))
            # NOTE: The [0] index is a hack since the test is fixed to use
            # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
            # be upgraded in the future to support multiple optimizers
            assert len(split_optimizer_states[table_index][0]) == num_ids
            opt = split_optimizer_states[table_index][0]
            new_ref_weight = torch.addcdiv(
                emb_r_w.float(),
                value=-lr,
                tensor1=emb_r_w_g,
                tensor2=opt.float()
                .sqrt_()
                .add_(eps)
                .view(
                    num_ids,
                    1,
                )
                .cuda(),
            ).cpu()

            emb_w = (
                emb_state_dict_list[table_index]
                .narrow(0, 0, bucket_asc_ids_list[table_index].size(0))
                .float()
            )
            torch.testing.assert_close(
                emb_w,
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_strategies,
        num_buckets=st.integers(min_value=10, max_value=15),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_state_dict_w_backend_return_whole_row(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        metaheader_dim = 16 // (weights_precision.bit_rate() // 8)  # 8-byte metaheader
        opt_dim = 4 // (weights_precision.bit_rate() // 8)  # 4-byte optimizer state

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            backend_type=BackendType.DRAM,
            enable_optimizer_offloading=True,
            backend_return_whole_row=True,
        )

        # Generate inputs
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
        )

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )
        split_optimizer_states = emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False
        )
        table_input_id_range = []
        for t, row in enumerate(Es):
            bucket_id_start = bucket_offsets[t][0]
            bucket_id_end = bucket_offsets[t][1]
            bucket_size = bucket_sizes[t]
            table_input_id_range.append(
                (
                    min(bucket_id_start * bucket_size, row),
                    min(bucket_id_end * bucket_size, row),
                )
            )
            # since we use ref_emb in dense format, the rows start from id 0
            self.assertEqual(table_input_id_range[-1][0], 0)

        """
        validate optimizer states
        """
        opt_validated = []
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_emb = emb_ref[f].weight.grad.float().to_dense().pow(2).cpu()
            ref_optimizer_state = ref_emb.mean(dim=1)[
                table_input_id_range[t][0] : min(
                    table_input_id_range[t][1], emb_ref[f].weight.size(0)
                )
            ]
            # pyre-fixme[16]: Undefined attribute: `Optional` has no attribute `__getitem__`.
            ref_kv_opt = ref_optimizer_state[bucket_asc_ids_list[t]].view(-1)
            opt = (
                # NOTE: The [0] index is a hack since the test is fixed to use
                # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
                # be upgraded in the future to support multiple optimizers
                split_optimizer_states[t][0]
                .narrow(0, 0, bucket_asc_ids_list[t].size(0))
                .view(-1)
                .view(torch.float32)
                .float()
            )
            opt_validated.append(opt.clone().detach())
            torch.testing.assert_close(
                opt,
                ref_kv_opt,
                atol=tolerance,
                rtol=tolerance,
            )

        for feature_index, table_index in self.get_physical_table_arg_indices_(
            emb.feature_table_map
        ):
            """
            validate bucket_asc_ids_list and num_active_id_per_bucket_list
            """
            bucket_asc_id = bucket_asc_ids_list[table_index]
            num_active_id_per_bucket = num_active_id_per_bucket_list[table_index]

            bucket_id_start = bucket_offsets[table_index][0]
            bucket_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_active_id_per_bucket.view(-1)
            )
            for bucket_idx, id_count in enumerate(num_active_id_per_bucket):
                bucket_id = bucket_idx + bucket_id_start
                active_id_cnt = 0
                for idx in range(
                    bucket_id_offsets[bucket_idx],
                    bucket_id_offsets[bucket_idx + 1],
                ):
                    # for chunk-based hashing
                    self.assertEqual(
                        bucket_id, bucket_asc_id[idx] // bucket_sizes[table_index]
                    )
                    active_id_cnt += 1
                self.assertEqual(active_id_cnt, id_count)

            """
            validate the whole embeddings rows (metaheader + weight + opt)
            """
            num_ids = len(bucket_asc_ids_list[table_index])
            emb_r_w = emb_ref[feature_index].weight[
                bucket_asc_ids_list[table_index].view(-1)
            ]
            emb_r_w_g = (
                emb_ref[feature_index]
                .weight.grad.float()
                .to_dense()[bucket_asc_ids_list[table_index].view(-1)]
            )
            self.assertLess(table_index, len(emb_state_dict_list))
            # NOTE: The [0] index is a hack since the test is fixed to use
            # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
            # be upgraded in the future to support multiple optimizers
            assert split_optimizer_states[table_index][0].size(0) == num_ids
            new_ref_weight = torch.addcdiv(
                emb_r_w.float(),
                value=-lr,
                tensor1=emb_r_w_g,
                tensor2=opt_validated[table_index]
                .clone()
                .sqrt_()
                .add_(eps)
                .view(
                    num_ids,
                    1,
                )
                .cuda(),
            ).cpu()

            emb_w = emb_state_dict_list[table_index].narrow(
                0, 0, bucket_asc_ids_list[table_index].size(0)
            )
            # Compare the opt part
            opt_extracted_from_emb_w = (
                emb_w[:, (metaheader_dim + D * 4) : (metaheader_dim + D * 4) + opt_dim]
                .view(torch.float32)
                .view(-1)
            )
            torch.testing.assert_close(
                opt_extracted_from_emb_w,
                opt_validated[table_index],
                atol=tolerance,
                rtol=tolerance,
            )

            # Copmare the id part
            id_extracted_from_emb_w = (
                emb_w[:, 0 : metaheader_dim // 2].view(torch.int64).view(-1)
            )
            torch.testing.assert_close(
                id_extracted_from_emb_w,
                bucket_asc_ids_list[table_index].view(-1),
                atol=tolerance,
                rtol=tolerance,
            )

            # Compare the weight part
            torch.testing.assert_close(
                emb_w[:, metaheader_dim : metaheader_dim + D * 4].float(),
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(**default_strategies)
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_fetch_from_l1_sp_w_row_ids_weight(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
        ) = self.generate_ssd_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
        )

        Es = [emb.embedding_specs[t][0] for t in range(T)]
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
        )

        updated_weights = torch.zeros(
            indices.numel(),
            emb.max_D,
            device=emb.current_device,
            dtype=emb.weights_precision.as_dtype(),
        )
        linearized_indices = []
        for f, idxes in enumerate(indices_list):
            linearized_indices.append(idxes.flatten().add(emb.hash_size_cumsum[f]))
        linearized_indices_tensor = torch.cat(linearized_indices)

        def copy_weights_hook(
            _grad: torch.Tensor,
            updated_weights: torch.Tensor,
            emb: SSDTableBatchedEmbeddingBags,
            row_ids: torch.Tensor,
        ) -> None:
            if row_ids.numel() != 0:
                updates_list, _mask = emb.fetch_from_l1_sp_w_row_ids(row_ids=row_ids)
                # The function now returns a list of tensors, but for weights we expect only one tensor
                updated_weights.copy_(updates_list[0])

        emb.register_backward_hook_before_eviction(
            lambda grad: copy_weights_hook(
                grad,
                updated_weights,
                emb,
                linearized_indices_tensor,
            )
        )

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare optimizer states
        split_optimizer_states = emb.split_optimizer_states()
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_optimizer_state = emb_ref[f].weight.grad.float().to_dense().pow(2)
            torch.testing.assert_close(
                # NOTE: The [0] index is a hack since the test is fixed to use
                # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
                # be upgraded in the future to support multiple optimizers
                split_optimizer_states[t][0].float(),
                ref_optimizer_state.mean(dim=1),
                atol=tolerance,
                rtol=tolerance,
            )

        # Compare weights
        emb.flush()

        cursor = 0
        emb_test = emb.debug_split_embedding_weights()
        for f, t in enumerate(emb.feature_table_map):
            row_idxes = indices_list[f]
            local_idxes = row_idxes.flatten()
            weights_per_tb = updated_weights[cursor : cursor + local_idxes.numel()]
            cursor += local_idxes.numel()

            if weights_precision == SparseType.FP16:
                # Round the reference weight the same way that TBE does
                weights_per_tb = weights_per_tb.half().float()

            # check only the updated rows
            torch.testing.assert_close(
                emb_test[t][local_idxes.cpu()].float().cuda(),
                weights_per_tb.float().cuda(),
                atol=tolerance,
                rtol=tolerance,
            )

    @given(**default_strategies)
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_fetch_from_l1_sp_w_row_ids_opt_only(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
    ) -> None:

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            enable_optimizer_offloading=True,
        )

        # Generate inputs
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
        )

        updated_opt_states = torch.zeros(
            indices.numel(),
            1,
            device=emb.current_device,
            # NOTE: This is a hack to keep fetch_from_l1_sp_w_row_ids unit test
            # working until it is upgraded to support optimizers with multiple
            # states and dtypes
            dtype=torch.float32,
        )
        linearized_indices = []
        for f, idxes in enumerate(indices_list):
            linearized_indices.append(idxes.flatten().add(emb.hash_size_cumsum[f]))
        linearized_indices_tensor = torch.cat(linearized_indices)

        def copy_opt_states_hook(
            _grad: torch.Tensor,
            updated_opt_states: torch.Tensor,
            emb: SSDTableBatchedEmbeddingBags,
            row_ids: torch.Tensor,
        ) -> None:
            if row_ids.numel() != 0:
                updates_list, _mask = emb.fetch_from_l1_sp_w_row_ids(
                    row_ids=row_ids, only_get_optimizer_states=True
                )
                # The function now returns a list of tensors
                # for EXACT_ROWWISE_ADAGRAD optimizer states we take the first one
                updated_opt_states.copy_(updates_list[0])

        emb.register_backward_hook_before_eviction(
            lambda grad: copy_opt_states_hook(
                grad,
                updated_opt_states,
                emb,
                linearized_indices_tensor,
            )
        )

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        # Compare emb state dict with expected values from nn.EmbeddingBag
        _emb_state_dict_list, bucket_asc_ids_list, _num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )
        assert bucket_asc_ids_list is not None
        split_optimizer_states = emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False
        )
        table_input_id_range = []
        for t, row in enumerate(Es):
            bucket_id_start = bucket_offsets[t][0]
            bucket_id_end = bucket_offsets[t][1]
            bucket_size = bucket_sizes[t]
            table_input_id_range.append(
                (
                    min(bucket_id_start * bucket_size, row),
                    min(bucket_id_end * bucket_size, row),
                )
            )
            # since we use ref_emb in dense format, the rows start from id 0
            self.assertEqual(table_input_id_range[-1][0], 0)

        cursor = 0
        tolerance = 1.0e-4
        # Compare optimizer states
        for f, t in enumerate(emb.feature_table_map):
            row_idxes = indices_list[f]
            local_idxes = row_idxes.flatten()
            value_to_index = {
                v.item(): i for i, v in enumerate(bucket_asc_ids_list[t].flatten())
            }
            indices = torch.tensor([value_to_index[v.item()] for v in local_idxes])
            opt_states_per_tb = updated_opt_states[
                cursor : cursor + local_idxes.numel()
            ].flatten()
            if local_idxes.numel() == 0:
                continue
            cursor += local_idxes.numel()

            torch.testing.assert_close(
                # NOTE: The [0] index is a hack since the test is fixed to use
                # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
                # be upgraded in the future to support multiple optimizers
                split_optimizer_states[t][0][indices].float(),
                opt_states_per_tb.cpu().float(),
                atol=tolerance,
                rtol=tolerance,
            )
