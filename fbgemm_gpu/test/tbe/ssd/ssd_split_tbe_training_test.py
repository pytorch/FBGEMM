# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]

import unittest
from typing import List, Optional, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.utils import b_indices, get_table_batched_offsets_from_dense
from hypothesis import assume, given, settings, Verbosity

from .. import common  # noqa E402
from ..common import open_source


if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, running_on_github
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, running_on_github


MAX_EXAMPLES = 40


@unittest.skipIf(*running_on_github)
@unittest.skipIf(*gpu_unavailable)
class SSDSplitTableBatchedEmbeddingsTest(unittest.TestCase):
    def get_physical_table_arg_indices_(self, feature_table_map: List[int]):
        """
        Get the physical table arg indices for the reference and TBE.  The
        first element in each tuple is for accessing the reference embedding
        list.  The second element is for accessing TBE data.

        Example:
            feature_table_map = [0, 1, 2, 2, 3, 4]
            This function returns [(0, 0), (1, 1), (2, 2), (4, 3), (5, 4)]
        """
        ref_arg_indices = []
        test_arg_indices = []
        prev_t = -1
        for f, t in enumerate(feature_table_map):
            # Only get the physical tables
            if prev_t != t:
                prev_t = t
                ref_arg_indices.append(f)
                test_arg_indices.append(t)
        return zip(ref_arg_indices, test_arg_indices)

    @given(
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd(self, weights_precision: SparseType) -> None:
        import tempfile

        E = int(1e4)
        D = 128
        N = 100
        indices = torch.as_tensor(np.random.choice(E, replace=False, size=(N,)))
        weights = torch.randn(N, D, dtype=weights_precision.as_dtype())
        output_weights = torch.empty_like(weights)
        count = torch.tensor([N])

        feature_table_map = list(range(1))
        emb = SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, D)],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=1,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            weights_precision=weights_precision,
        )
        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()
        assert (output_weights <= 0.1).all().item()
        assert (output_weights >= -0.1).all().item()

        emb.ssd_db.set_cuda(indices, weights, count, 1)
        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()
        torch.testing.assert_close(weights, output_weights)

    def generate_inputs_(
        self,
        B: int,
        L: int,
        Es: List[int],
        feature_table_map: List[int],
        weights_precision: SparseType = SparseType.FP32,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Generate indices and per sample weights
        """
        T = len(feature_table_map)

        # Generate random indices and per sample weights
        indices_list = [
            torch.randint(low=0, high=Es[t], size=(B, L)).cuda()
            for t in feature_table_map
        ]
        per_sample_weights_list = [torch.randn(size=(B, L)).cuda() for _ in range(T)]

        # Concat inputs for SSD TBE
        indices = torch.cat([indices.view(1, B, L) for indices in indices_list], dim=0)
        per_sample_weights = torch.cat(
            [
                per_sample_weights.view(1, B, L)
                for per_sample_weights in per_sample_weights_list
            ],
            dim=0,
        )
        (indices, offsets) = get_table_batched_offsets_from_dense(indices)

        return (
            indices_list,
            per_sample_weights_list,
            indices.cuda(),
            offsets.cuda(),
            per_sample_weights.contiguous().view(-1).cuda(),
        )

    def generate_ssd_tbes(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        lr: float = 0.01,  # from SSDTableBatchedEmbeddingBags
        eps: float = 1.0e-8,  # from SSDTableBatchedEmbeddingBags
        ssd_shards: int = 1,  # from SSDTableBatchedEmbeddingBags
        optimizer: OptimType = OptimType.EXACT_ROWWISE_ADAGRAD,
        cache_set_scale: float = 1.0,
        pooling_mode: bool = PoolingMode.SUM,
        weights_precision: SparseType = SparseType.FP32,
        output_dtype: SparseType = SparseType.FP32,
        stochastic_rounding: bool = True,
        share_table: bool = False,
    ) -> Tuple[SSDTableBatchedEmbeddingBags, List[torch.nn.EmbeddingBag]]:
        """
        Generate embedding modules (i,e., SSDTableBatchedEmbeddingBags and
        torch.nn.EmbeddingBags)
        """
        import tempfile

        torch.manual_seed(42)
        E = int(10**log_E)
        D = D * 4
        Ds = [D] * T
        Es = [E] * T

        if pooling_mode == PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
        elif pooling_mode == PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
        elif pooling_mode == PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
        else:
            # This proves that we have exhaustively checked all PoolingModes
            raise RuntimeError("Unknown PoolingMode!")

        # Generate torch EmbeddingBag
        if do_pooling:
            emb_ref = [
                torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True).cuda()
                for (E, D) in zip(Es, Ds)
            ]
        else:
            emb_ref = [
                torch.nn.Embedding(E, D, sparse=True).cuda() for (E, D) in zip(Es, Ds)
            ]

        # Cast type
        if weights_precision == SparseType.FP16:
            emb_ref = [emb.half() for emb in emb_ref]

        # Construct feature_table_map
        feature_table_map = list(range(T))
        table_to_replicate = -1
        if share_table:
            # autograd with shared embedding only works for exact
            table_to_replicate = T // 2
            # pyre-fixme[6]: For 2nd param expected `Embedding` but got
            #  `Union[Embedding, EmbeddingBag]`.
            feature_table_map.insert(table_to_replicate, table_to_replicate)
            emb_ref.insert(table_to_replicate, emb_ref[table_to_replicate])

        cache_sets = max(int(max(T * B * L, 1) * cache_set_scale), 1)

        # Generate TBE SSD
        emb = SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=cache_sets,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            learning_rate=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            optimizer=optimizer,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            stochastic_rounding=stochastic_rounding,
        ).cuda()

        # Initialize TBE SSD weights
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            emb_r = emb_ref[f]
            emb_r.weight.data.uniform_(-2.0, 2.0)
            emb.ssd_db.set_cuda(
                torch.arange(t * E, (t + 1) * E).to(torch.int64),
                emb_r.weight.cpu(),
                torch.as_tensor([E]),
                t,
            )

        # Convert back to float (to make sure that accumulation is done
        # in FP32 -- like TBE)
        if weights_precision == SparseType.FP16:
            emb_ref = [emb.float() for emb in emb_ref]

        return emb, emb_ref

    def concat_ref_tensors(
        self,
        tensors: List[torch.Tensor],
        do_pooling: bool,
        B: int,
        D: int,
    ) -> torch.Tensor:
        if do_pooling:
            return torch.cat([t.view(B, -1) for t in tensors], dim=1)
        return torch.cat(tensors, dim=0).view(-1, D)

    def execute_ssd_forward_(
        self,
        emb: SSDTableBatchedEmbeddingBags,
        emb_ref: List[torch.nn.EmbeddingBag],
        indices_list: List[torch.Tensor],
        per_sample_weights_list: List[torch.Tensor],
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: torch.Tensor,
        B: int,
        L: int,
        weighted: bool,
        tolerance: Optional[float] = None,
        it: int = -1,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Execute the forward functions of SSDTableBatchedEmbeddingBags and
        torch.nn.EmbeddingBag and compare outputs
        """
        assert len(emb_ref) == len(indices_list)
        do_pooling = emb.pooling_mode != PoolingMode.NONE
        # Execute torch EmbeddingBag forward
        output_ref_list = (
            [
                b_indices(emb_, indices, do_pooling=do_pooling)
                for (emb_, indices) in zip(emb_ref, indices_list)
            ]
            if not weighted
            else [
                b_indices(
                    emb_,
                    indices,
                    per_sample_weights=per_sample_weights.view(-1),
                    do_pooling=do_pooling,
                )
                for (emb_, indices, per_sample_weights) in zip(
                    emb_ref, indices_list, per_sample_weights_list
                )
            ]
        )

        output_ref = self.concat_ref_tensors(
            output_ref_list,
            do_pooling,
            B,
            emb.embedding_specs[0][1],
        )

        # Execute TBE SSD forward
        output = (
            emb(indices, offsets)
            if not weighted
            else emb(indices, offsets, per_sample_weights)
        )

        out_dtype = output.dtype
        # Cast the ref output type the output types do not match between ref
        # and test
        if output_ref.dtype != out_dtype:
            output_ref_list = [out.to(out_dtype) for out in output_ref_list]
            output_ref = output_ref.to(out_dtype)

        # Set tolerance
        tolerance = (
            (
                1.0e-5
                if emb_ref[0].weight.dtype == torch.float and out_dtype == torch.float
                else 8.0e-3
            )
            if tolerance is None
            else tolerance
        )

        # Compare outputs
        torch.testing.assert_close(
            output.float(),
            output_ref.float(),
            atol=tolerance,
            rtol=tolerance,
        )
        return output_ref_list, output

    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
        cache_set_scale=st.sampled_from([0.0, 0.005, 1]),
        pooling_mode=st.sampled_from(
            [PoolingMode.NONE, PoolingMode.SUM, PoolingMode.MEAN]
        ),
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        share_table=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_forward(
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
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)

        # Generate embedding modules
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
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
        )

        # Generate inputs
        Es = [emb.embedding_specs[t][0] for t in range(T)]
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
        ) = self.generate_inputs_(
            B, L, Es, emb.feature_table_map, weights_precision=weights_precision
        )

        # Execute forward
        self.execute_ssd_forward_(
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
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
        cache_set_scale=st.sampled_from([0.0, 0.005, 1]),
        pooling_mode=st.sampled_from(
            [PoolingMode.NONE, PoolingMode.SUM, PoolingMode.MEAN]
        ),
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        share_table=st.booleans(),
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
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)

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
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
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
        )

        # Generate output gradient
        output_grad_list = [torch.randn_like(out) for out in output_ref_list]

        # Execute torch EmbeddingBag backward
        [out.backward(grad) for (out, grad) in zip(output_ref_list, output_grad_list)]

        do_pooling = pooling_mode != PoolingMode.NONE
        grad_test = self.concat_ref_tensors(
            output_grad_list,
            do_pooling,
            B,
            D * 4,
        )

        # Execute TBE SSD backward
        output.backward(grad_test)

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare optimizer states
        split_optimizer_states = [s for (s,) in emb.debug_split_optimizer_states()]
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_optimizer_state = emb_ref[f].weight.grad.float().to_dense().pow(2)
            torch.testing.assert_close(
                split_optimizer_states[t].float(),
                ref_optimizer_state.mean(dim=1),
                atol=tolerance,
                rtol=tolerance,
            )

        # Compare weights
        emb.flush()

        emb_test = emb.debug_split_embedding_weights()
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            emb_r = emb_ref[f]
            new_ref_weight = torch.addcdiv(
                emb_r.weight.float(),
                value=-lr,
                tensor1=emb_r.weight.grad.float().to_dense(),
                tensor2=split_optimizer_states[t]
                .float()
                .sqrt_()
                .add_(eps)
                .view(Es[t], 1),
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
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
        cache_set_scale=st.sampled_from([0.0, 0.005, 1]),
        pooling_mode=st.sampled_from(
            [PoolingMode.NONE, PoolingMode.SUM, PoolingMode.MEAN]
        ),
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        share_table=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_cache(
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
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)

        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        torch.manual_seed(42)

        # Generate embedding modules
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
            # Disable stochastic rounding because error is too large when
            # running for many iterations. This should be OK for testing the
            # functionality of the cache
            stochastic_rounding=False,
            share_table=share_table,
        )

        optimizer_states_ref = [
            s.clone().float() for (s,) in emb.debug_split_optimizer_states()
        ]

        Es = [emb.embedding_specs[t][0] for t in range(T)]

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        for it in range(10):
            (
                indices_list,
                per_sample_weights_list,
                indices,
                offsets,
                per_sample_weights,
            ) = self.generate_inputs_(
                B,
                L,
                Es,
                emb.feature_table_map,
                weights_precision=weights_precision,
            )
            assert emb.timestep == it

            emb.prefetch(indices, offsets)

            linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                emb.hash_size_cumsum,
                indices,
                offsets,
            )

            # Verify that prefetching twice avoids any actions.
            (
                _,
                _,
                _,
                actions_count_gpu,
                _,
                _,
                _,
                _,
            ) = torch.ops.fbgemm.ssd_cache_populate_actions(  # noqa
                linear_cache_indices,
                emb.total_hash_size,
                emb.lxu_cache_state,
                emb.timestep,
                0,  # prefetch_dist
                emb.lru_state,
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
                tolerance=tolerance,
                it=it,
            )

            # Generate output gradient
            output_grad_list = [torch.randn_like(out) for out in output_ref_list]

            # Zero out weight grad
            for f, _ in self.get_physical_table_arg_indices_(emb.feature_table_map):
                emb_ref[f].weight.grad = None

            # Execute torch EmbeddingBag backward
            for out, grad in zip(output_ref_list, output_grad_list):
                out.backward(grad)

            do_pooling = pooling_mode != PoolingMode.NONE
            grad_test = self.concat_ref_tensors(
                output_grad_list,
                do_pooling,
                B,
                D * 4,
            )

            # Execute TBE SSD backward
            output.backward(grad_test)

            # Compare optimizer states
            split_optimizer_states = [s for (s,) in emb.debug_split_optimizer_states()]
            for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
                optim_state_r = optimizer_states_ref[t]
                optim_state_t = split_optimizer_states[t]
                emb_r = emb_ref[f]

                # pyre-fixme[16]: Optional type has no attribute `float`.
                optim_state_r.add_(
                    emb_r.weight.grad.float().to_dense().pow(2).mean(dim=1)
                )
                torch.testing.assert_close(
                    optim_state_t.float(),
                    optim_state_r,
                    atol=tolerance,
                    rtol=tolerance,
                )

                new_ref_weight = torch.addcdiv(
                    emb_r.weight.float(),
                    value=-lr,
                    tensor1=emb_r.weight.grad.float().to_dense(),
                    tensor2=optim_state_t.float().sqrt().add(eps).view(Es[t], 1),
                )

                if weights_precision == SparseType.FP16:
                    # Round the reference weight the same way that
                    # TBE does
                    new_ref_weight = new_ref_weight.half().float()
                    assert new_ref_weight.dtype == emb_r.weight.dtype

                emb_r.weight.data.copy_(new_ref_weight)

        # Compare weights
        emb.flush()
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            weight_r = emb_ref[f].weight.float()
            weight_t = emb.debug_split_embedding_weights()[t].float().cuda()
            torch.testing.assert_close(
                weight_t,
                weight_r,
                atol=tolerance,
                rtol=tolerance,
            )
