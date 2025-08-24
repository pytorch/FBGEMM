# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]

import unittest
from typing import Any, Dict

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
    print_different_rows,
    SSDSplitTableBatchedEmbeddingsTestCommon,
    VIRTUAL_TABLE_ROWS,
)

default_st: Dict[str, Any] = default_strategies | {
    "m1_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
    "m2_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
    "num_buckets": st.integers(min_value=10, max_value=15),
}


@unittest.skipIf(*running_in_oss)
@unittest.skipIf(*gpu_unavailable)
class SSDSplitTBEPartialRowwiseAdamTest(SSDSplitTableBatchedEmbeddingsTestCommon):
    @given(
        **default_st,
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_backward_partial_rowwise_adam(
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
        m1_dtype: SparseType,
        m2_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        backend_type: BackendType,
        **kwargs: Any,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        # VBE is currently not supported for PARTIAL_ROWWISE_ADAM optimizer
        assume(not mixed_B)

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01

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
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            backend_type=backend_type,
            optimizer_state_dtypes={
                "momentum1": m1_dtype,
                "momentum2": m2_dtype,
            },
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

        tolerance = 1.0e-2

        emb_test_weights = emb.debug_split_embedding_weights()
        split_optimizer_states = self.split_optimizer_states_(emb)

        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (m1, m2) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = (ref_grad.pow(2).mean(dim=1)) * (1.0 - beta2)
            self.assert_close_(m2, m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            self.assert_close_(m1, m1_ref)

            # Bias corrections
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            v_hat_t = v_hat_t.view(v_hat_t.numel(), 1)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Weight update
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            if weights_precision == SparseType.FP16:
                # Round the reference weight the same way that TBE does
                ref_weights_updated = ref_weights_updated.half().float()

            # Compare weights
            torch.testing.assert_close(
                emb_test_weights[t].float().cuda(),
                ref_weights_updated.cuda(),
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_st,
        bulk_init_chunk_size=st.sampled_from([0, 204800]),
        lazy_bulk_init_enabled=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_emb_state_dict_partial_rowwise_adam(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        m1_dtype: SparseType,
        m2_dtype: SparseType,
        bulk_init_chunk_size: int,
        lazy_bulk_init_enabled: bool,
        **kwargs: Any,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01

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
            weighted,  # weighted
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
            cache_set_scale=0.2,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=True,
            bulk_init_chunk_size=bulk_init_chunk_size,
            lazy_bulk_init_enabled=lazy_bulk_init_enabled,
            optimizer_state_dtypes={
                "momentum1": m1_dtype,
                "momentum2": m2_dtype,
            },
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

        tolerance = 1.0e-2

        split_optimizer_states = self.split_optimizer_states_(emb)

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict, _, _, _ = emb.split_embedding_weights(no_snapshot=False)
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (
                m1,
                m2,
            ) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = (ref_grad.pow(2).mean(dim=1)) * (1.0 - beta2)
            self.assert_close_(m2, m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            self.assert_close_(m1, m1_ref)

            # Bias corrections
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            v_hat_t = v_hat_t.view(v_hat_t.numel(), 1)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Weight update
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            # Compare weights
            torch.testing.assert_close(
                # pyre-fixme [16]
                emb_state_dict[t].full_tensor().float(),
                ref_weights_updated,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_st,
        enable_optimizer_offloading=st.booleans(),
        backend_type=st.just(BackendType.DRAM),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_emb_state_dict_partial_rowwise_adam(
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
        m1_dtype: SparseType,
        m2_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        enable_optimizer_offloading: bool,
        backend_type: BackendType,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        # VBE is currently not supported for PARTIAL_ROWWISE_ADAM optimizer
        assume(not mixed_B)
        # Don't stimulate boundary check cases
        trigger_bounds_check = False

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01

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
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_type=backend_type,
            optimizer_state_dtypes={
                "momentum1": m1_dtype,
                "momentum2": m2_dtype,
            },
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

        split_optimizer_states = []

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )

        for s in emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        ):
            split_optimizer_states.append(s)

        # Compare optimizer states
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (m1, m2) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = (ref_grad.pow(2).mean(dim=1)) * (1.0 - beta2)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            # pyre-ignore [16]
            m2_ref = m2_ref[bucket_asc_ids_list[t].view(-1)]
            self.assert_close_(m2, m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            m1_ref = m1_ref[bucket_asc_ids_list[t].view(-1)]

            # Print which rows are different between m1 and m1_ref
            print_different_rows(
                m1, m1_ref, atol=1.0e-2, rtol=1.0e-2, name1="m1", name2="m1_ref"
            )

            self.assert_close_(m1, m1_ref)
            ####################################################################
            # Compare weight values
            ####################################################################

            # Re-index the weights according to bucket ids
            ref_weights = ref_weights[bucket_asc_ids_list[t].view(-1)]

            # Bias corrections
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            v_hat_t = v_hat_t.view(v_hat_t.numel(), 1)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Manually update the ref weights
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            # Fetch the updated weights from SSDTableBatchedEmbeddingBags
            emb_w = (
                emb_state_dict_list[t]
                .narrow(0, 0, bucket_asc_ids_list[t].size(0))
                .float()
            )

            # Compare weights
            tolerance = 1.0e-2
            torch.testing.assert_close(
                emb_w,
                ref_weights_updated,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_st,
        enable_optimizer_offloading=st.just(True),
        backend_type=st.sampled_from([BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_emb_state_dict_partial_rowwise_adam_whole_row(
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
        m1_dtype: SparseType,
        m2_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        enable_optimizer_offloading: bool,
        backend_type: BackendType,
        **kwargs: Any,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        # VBE is currently not supported for PARTIAL_ROWWISE_ADAM optimizer
        assume(not mixed_B)
        # Don't stimulate boundary check cases
        trigger_bounds_check = False

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01
        metaheader_dim = 16 // (
            weights_precision.as_dtype().itemsize
        )  # 16-byte metaheader

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
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_type=backend_type,
            backend_return_whole_row=True,
            optimizer_state_dtypes={
                "momentum1": m1_dtype,
                "momentum2": m2_dtype,
            },
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

        split_optimizer_states = []

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, _, _ = emb.split_embedding_weights(
            no_snapshot=False, should_flush=True
        )

        for s in emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        ):
            split_optimizer_states.append(s)

        # Compare optimizer states
        table_offset = 0
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (m1, m2) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = (ref_grad.pow(2).mean(dim=1)) * (1.0 - beta2)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            # pyre-ignore [16]
            m2_ref = m2_ref[bucket_asc_ids_list[t].view(-1)]
            m2_tensor = m2.full_tensor().view(m2_dtype.as_dtype())
            if m2_tensor.shape[1] > 1:
                # HACK: the KVTensorWrapper does not support dtypes other than weight dtype,
                # so when the optimizer state dtype is smaller than weight dtype, we'll save more
                # data than needed. Before we support the optimizer dtype from backend, we workaround the
                # verification in test:
                # the m2 dtype could be smaller than weight dtype, so we need to get the first column as the
                # m2 value, and then convert it to the same dtype as m2_ref and then compare with it.
                m2_tensor = m2_tensor[:, 0]
            # Compare the weight part
            self.assert_close_(m2_tensor.view(-1), m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            m1_ref = m1_ref[bucket_asc_ids_list[t].view(-1)]
            m1_tensor = m1.full_tensor().view(m1_dtype.as_dtype())

            # Print which rows are different between m1 and m1_ref
            print_different_rows(
                m1_tensor, m1_ref, atol=1.0e-2, rtol=1.0e-2, name1="m1", name2="m1_ref"
            )

            self.assert_close_(m1_tensor, m1_ref)

            ####################################################################
            # Compare weight values
            ####################################################################
            # Re-index the weights according to bucket ids
            ref_weights = ref_weights[bucket_asc_ids_list[t].view(-1)]
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            v_hat_t = v_hat_t.view(v_hat_t.numel(), 1)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Manually update the ref weights
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            """
            validate the whole embeddings rows (metaheader + weight + m2 + m1)
            """

            emb_w = emb_state_dict_list[t].narrow(0, 0, bucket_asc_ids_list[t].size(0))
            emb_d = emb.embedding_specs[t][1]

            # Lookup the byte offsets for each optimizer state
            optimizer_state_byte_offsets = emb.optimizer.byte_offsets_along_row(
                emb_d,
                emb.weights_precision,
                emb.optimizer_state_dtypes,
            )

            m2_start, m2_end = optimizer_state_byte_offsets["momentum2"]
            m1_start, m1_end = optimizer_state_byte_offsets["momentum1"]
            meta_bytes = metaheader_dim * weights_precision.as_dtype().itemsize
            m2_from_emb_w = (
                emb_w.view(dtype=torch.uint8)[
                    :,
                    meta_bytes + m2_start : meta_bytes + m2_end,
                ]
                .view(m2_dtype.as_dtype())
                .view(-1)
            )
            m1_from_emb_w = emb_w.view(dtype=torch.uint8)[
                :,
                meta_bytes + m1_start : meta_bytes + m1_end,
            ].view(m1_dtype.as_dtype())
            self.assert_close_(m2_from_emb_w, m2_ref)
            self.assert_close_(m1_from_emb_w, m1_ref)

            # Copmare the id part
            id_extracted_from_emb_w = (
                emb_w[:, 0 : metaheader_dim // 2].view(torch.int64).view(-1)
            )
            self.assert_close_(
                id_extracted_from_emb_w, bucket_asc_ids_list[t].view(-1) + table_offset
            )
            # Compare the weight part
            tolerance = 1.0e-2
            torch.testing.assert_close(
                emb_w[:, metaheader_dim : metaheader_dim + emb_d].float(),
                ref_weights_updated,
                atol=tolerance,
                rtol=tolerance,
            )

            table_offset += VIRTUAL_TABLE_ROWS

    @given(**default_st)
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_fetch_from_l1_sp_w_row_ids_partial_rowwise_adam(
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
        **kwargs: Any,
    ) -> None:

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.999
        weight_decay = 0.01

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not mixed_B)
        assume(not weighted or pooling_mode == PoolingMode.SUM)

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
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
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
            mixed_B=False,
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
        )

        # For PARTIAL_ROWWISE_ADAM, we need to handle two optimizer states (m1 and m2)
        updated_m1 = torch.zeros(
            indices.numel(),
            emb.max_D,
            device=emb.current_device,
            dtype=torch.float32,
        )

        updated_m2 = torch.zeros(
            indices.numel(),
            1,
            device=emb.current_device,
            dtype=torch.float32,
        )

        linearized_indices = []
        for f, idxes in enumerate(indices_list):
            linearized_indices.append(idxes.flatten().add(emb.hash_size_cumsum[f]))
        linearized_indices_tensor = torch.cat(linearized_indices)

        def copy_opt_states_hook(
            _grad: torch.Tensor,
            updated_m1: torch.Tensor,
            updated_m2: torch.Tensor,
            emb: SSDTableBatchedEmbeddingBags,
            row_ids: torch.Tensor,
        ) -> None:
            if row_ids.numel() != 0:
                updates_list, _mask = emb.fetch_from_l1_sp_w_row_ids(
                    row_ids=row_ids, only_get_optimizer_states=True
                )
                # The function now returns a list of tensors for optimizer states
                # For PARTIAL_ROWWISE_ADAM, we expect two tensors: m1 and m2
                updated_m1.copy_(updates_list[1])
                updated_m2.copy_(updates_list[0])

        emb.register_backward_hook_before_eviction(
            lambda grad: copy_opt_states_hook(
                grad,
                updated_m1,
                updated_m2,
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
            m1_states_per_tb = updated_m1[cursor : cursor + local_idxes.numel()]
            m2_states_per_tb = updated_m2[cursor : cursor + local_idxes.numel()]

            if local_idxes.numel() == 0:
                continue
            cursor += local_idxes.numel()

            # For PARTIAL_ROWWISE_ADAM, we have two optimizer states: m1 and m2
            torch.testing.assert_close(
                split_optimizer_states[t][0][indices].float(),
                m1_states_per_tb.cpu().float(),
                atol=tolerance,
                rtol=tolerance,
            )

            torch.testing.assert_close(
                split_optimizer_states[t][1][indices].float(),
                m2_states_per_tb.cpu().flatten().float(),
                atol=tolerance,
                rtol=tolerance,
            )
