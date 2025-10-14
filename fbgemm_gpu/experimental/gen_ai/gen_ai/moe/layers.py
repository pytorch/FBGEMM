# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional, Union

import torch

from fairscale.nn.model_parallel.initialize import get_model_parallel_world_size

from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import triton_quantize_fp8_row
from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import (
    grouped_gemm,
    grouped_gemm_fp8_rowwise,
)
from fbgemm_gpu.experimental.gen_ai.moe.activation import silu_mul, silu_mul_quant
from fbgemm_gpu.experimental.gen_ai.moe.gather_scatter import (
    gather_scale_dense_tokens,
    gather_scale_quant_dense_tokens,
    scatter_add_dense_tokens,
    scatter_add_padded_tokens,
)
from fbgemm_gpu.experimental.gen_ai.moe.shuffling import (
    combine_shuffling,
    split_shuffling,
)
from pyre_extensions import none_throws
from torch.distributed import get_rank, ProcessGroup

try:
    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu import open_source
except Exception:
    open_source: bool = False

if open_source:
    torch.ops.load_library(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "fbgemm_gpu_experimental_gen_ai.so",
        )
    )
    torch.classes.load_library(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "fbgemm_gpu_experimental_gen_ai.so",
        )
    )
else:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:index_shuffling_ops"
    )

if torch.cuda.is_available():
    index_shuffling = torch.ops.fbgemm.index_shuffling  # noqa F401
else:
    index_shuffling = None


__all__ = ["MoEArgs", "BaselineMoE", "MetaShufflingMoE"]


@dataclass(frozen=True)
class MoEArgs:
    precision: str
    dim: int
    hidden_dim: int
    num_experts: int
    top_k: int
    mp_size: int
    ep_size: int
    mp_size_for_routed_experts: Optional[int]
    use_fast_accum: bool
    dedup_comm: bool

    @cached_property
    def num_local_experts(self) -> int:
        return self.num_experts // self.ep_size


INIT_METHODS_TYPE = Mapping[
    str,
    Callable[[torch.Tensor], Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]],
]


class ScaledParameter(torch.nn.Parameter):
    def __new__(
        cls,
        data: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> "ScaledParameter":
        return super().__new__(cls, data, False)

    def __init__(
        self,
        data: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ):
        self._scale: Optional[torch.Tensor] = scale

    @property
    def weights(self) -> torch.Tensor:
        return self.data

    @property
    def scales(self) -> torch.Tensor:
        assert self._scale is not None
        return self._scale

    @scales.setter
    def scales(self, s: torch.Tensor) -> None:
        self._scale = s

    @property
    def is_scaled(self) -> bool:
        return self._scale is not None


# Helper functions/modules to perform weights sharding and initialization.
def init_params(
    key: str,
    param: ScaledParameter,
    init_methods: INIT_METHODS_TYPE,
):
    if key in init_methods:
        ret = init_methods[key](param.data)
        if isinstance(ret, torch.Tensor):
            param.data = ret
        else:
            param.data, param.scales = ret
    else:
        torch.nn.init.kaiming_uniform_(param)


class Experts(torch.nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.dim: int = dim
        self.hidden_dim: int = hidden_dim

        self.dtype: torch.dtype = torch.get_default_dtype()
        self.divide_factor: int = get_model_parallel_world_size()

        assert self.dim % self.divide_factor == 0
        assert self.hidden_dim % self.divide_factor == 0

        self._w13: Optional[ScaledParameter] = None
        self._w2: Optional[ScaledParameter] = None

    @abstractmethod
    def build(self, init_methods: Optional[INIT_METHODS_TYPE] = None) -> "Experts":
        pass

    @property
    def w13(self) -> ScaledParameter:
        assert self._w13 is not None, "Parameters are not initialized!"
        return self._w13

    @property
    def w2(self) -> ScaledParameter:
        assert self._w2 is not None, "Parameters are not initialized!"
        return self._w2

    @property
    def is_fp8_rowwise(self) -> bool:
        return self.w13.dtype == torch.float8_e4m3fn


class RoutedExperts(Experts):
    def __init__(
        self,
        num_local_experts: int,
        dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__(dim, hidden_dim)

        self.num_local_experts: int = num_local_experts

    def build(
        self, init_methods: Optional[INIT_METHODS_TYPE] = None
    ) -> "RoutedExperts":
        init_methods = {} if init_methods is None else init_methods

        moe_w_in_eDF: ScaledParameter = ScaledParameter(
            torch.empty(
                self.num_local_experts,
                self.dim,
                self.hidden_dim // self.divide_factor,
                dtype=self.dtype,
            )
        )
        init_params("moe_w_in_eDF", moe_w_in_eDF, init_methods)

        moe_w_out_eFD: ScaledParameter = ScaledParameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_dim // self.divide_factor,
                self.dim,
                dtype=self.dtype,
            )
        )
        init_params("moe_w_out_eFD", moe_w_out_eFD, init_methods)

        moe_w_swiglu_eDF: ScaledParameter = ScaledParameter(
            torch.empty(
                self.num_local_experts,
                self.dim,
                self.hidden_dim // self.divide_factor,
                dtype=self.dtype,
            )
        )
        init_params("moe_w_swiglu_eDF", moe_w_swiglu_eDF, init_methods)

        assert (
            moe_w_in_eDF.dtype == moe_w_out_eFD.dtype
            and moe_w_in_eDF.dtype == moe_w_swiglu_eDF.dtype
        )
        assert (
            moe_w_in_eDF.is_scaled == moe_w_out_eFD.is_scaled
            and moe_w_in_eDF.is_scaled == moe_w_swiglu_eDF.is_scaled
        )

        self._w13 = ScaledParameter(
            data=torch.cat(
                [
                    moe_w_in_eDF,
                    moe_w_swiglu_eDF,
                ],
                dim=-1,
            )
            .transpose(1, 2)
            .contiguous(),
            scale=(
                torch.cat(
                    [
                        moe_w_in_eDF.scales,
                        moe_w_swiglu_eDF.scales,
                    ],
                    dim=-1,
                ).contiguous()
                if moe_w_in_eDF.is_scaled
                else None
            ),
        )

        del moe_w_in_eDF
        del moe_w_swiglu_eDF

        self._w2 = ScaledParameter(
            data=moe_w_out_eFD.transpose(1, 2).contiguous(),
            scale=(
                moe_w_out_eFD.scales.contiguous() if moe_w_out_eFD.is_scaled else None
            ),
        )

        del moe_w_out_eFD

        return self


class SharedExperts(Experts):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__(dim, hidden_dim)

    def build(
        self, init_methods: Optional[INIT_METHODS_TYPE] = None
    ) -> "SharedExperts":
        init_methods = {} if init_methods is None else init_methods

        w_in_shared_FD = ScaledParameter(
            torch.empty(
                (self.hidden_dim // self.divide_factor, self.dim), dtype=self.dtype
            )
        )
        init_params("w_in_shared_FD", w_in_shared_FD, init_methods)

        w_out_shared_DF = ScaledParameter(
            torch.empty(
                (self.dim, self.hidden_dim // self.divide_factor), dtype=self.dtype
            )
        )
        init_params("w_out_shared_DF", w_out_shared_DF, init_methods)

        w_swiglu_FD = ScaledParameter(
            torch.empty(
                (self.hidden_dim // self.divide_factor, self.dim), dtype=self.dtype
            )
        )
        init_params("w_swiglu_FD", w_swiglu_FD, init_methods)

        assert (w_in_shared_FD.dtype == w_out_shared_DF.dtype) and (
            w_in_shared_FD.dtype == w_swiglu_FD.dtype
        )
        assert (w_in_shared_FD.is_scaled == w_out_shared_DF.is_scaled) and (
            w_in_shared_FD.is_scaled == w_swiglu_FD.is_scaled
        )

        self._w13 = ScaledParameter(
            data=torch.cat(
                [
                    w_in_shared_FD,
                    w_swiglu_FD,
                ]
            ).contiguous(),
            scale=(
                torch.cat(
                    [
                        w_in_shared_FD.scales,
                        w_swiglu_FD.scales,
                    ]
                ).contiguous()
                if w_in_shared_FD.is_scaled
                else None
            ),
        )
        del w_in_shared_FD
        del w_swiglu_FD

        self._w2 = ScaledParameter(
            data=w_out_shared_DF.data.contiguous(),
            scale=(
                w_out_shared_DF.scales.contiguous()
                if w_out_shared_DF.is_scaled
                else None
            ),
        )
        del w_out_shared_DF

        return self


class BaselineMoE(torch.nn.Module):
    def __init__(
        self,
        ep_group: ProcessGroup,
        ep_mp_group: ProcessGroup,
        moe_args: MoEArgs,
    ) -> None:
        super().__init__()

        self.moe_args = moe_args
        self.mp_size: int = moe_args.mp_size
        self.ep_size: int = moe_args.ep_size
        self.ep_mp_size: int = (
            moe_args.mp_size
            if moe_args.mp_size_for_routed_experts is None
            else moe_args.mp_size_for_routed_experts
        )

        self.ep_rank: int = get_rank(ep_group)
        self.ep_mp_rank: int = get_rank(ep_mp_group)

        self.ep_mp_group: ProcessGroup = ep_mp_group
        self.ep_group: ProcessGroup = ep_group

        self.num_experts: int = moe_args.num_experts
        self.num_local_experts: int = none_throws(moe_args.num_local_experts)
        assert self.num_experts == self.num_local_experts * self.ep_size

        self.top_k: int = moe_args.top_k

        self.dtype: torch.dtype = torch.get_default_dtype()

        self._router_DE: Optional[ScaledParameter] = None
        self.routed_experts = RoutedExperts(
            moe_args.num_local_experts,
            moe_args.dim,
            moe_args.hidden_dim,
        )
        self.shared_experts = SharedExperts(
            moe_args.dim,
            moe_args.hidden_dim,
        )

    def build(self, init_methods: Optional[INIT_METHODS_TYPE] = None) -> "BaselineMoE":
        init_methods = {} if init_methods is None else init_methods

        router_DE = ScaledParameter(
            torch.empty(self.moe_args.dim, self.moe_args.num_experts, dtype=self.dtype)
        )
        init_params("router_DE", router_DE, init_methods)
        self._router_DE = router_DE

        self.routed_experts.build(init_methods)
        self.shared_experts.build(init_methods)
        return self

    @property
    def router_DE(self) -> ScaledParameter:
        assert self._router_DE is not None, "Parameters are not initialized!"
        return self._router_DE

    # User should overwrite this property
    @property
    def is_shared_fp8_rowwise(self) -> bool:
        return self.shared_experts.is_fp8_rowwise

    @property
    def is_routed_fp8_rowwise(self) -> bool:
        return self.routed_experts.is_fp8_rowwise

    @property
    def E(self) -> int:
        return self.num_experts

    @property
    def EG(self) -> int:
        return self.num_local_experts

    @property
    def K(self) -> int:
        return self.top_k

    def forward(self, x: torch.Tensor, use_static_shape: bool) -> torch.Tensor:
        with torch.no_grad():
            return self._forward(x, use_static_shape)

    def _forward(self, x: torch.Tensor, use_static_shape: bool) -> torch.Tensor:
        (B, T, D) = x.shape
        T *= B
        tokens = x.view(T, D)

        # Shared Experts
        shared_y = self._fake_quant(torch.mm, tokens, self.shared_experts.w13)
        shared_y0, shared_y1 = torch.chunk(shared_y, chunks=2, dim=-1)
        shared_z = shared_y0 * torch.sigmoid(shared_y0) * shared_y1
        shared_z = self._fake_quant(torch.mm, shared_z, self.shared_experts.w2)

        # Routing Scores
        E: int = self.E
        scores = torch.nn.functional.linear(tokens, self.router_DE.T)
        scores = torch.sigmoid(scores)
        assert scores.shape == (T, E)

        # Routing
        K: int = self.K
        topk_values, topk_indices = torch.topk(scores, K, dim=-1)
        assert topk_values.shape == (T, K)
        assert topk_indices.shape == (T, K)

        masked_scores = torch.zeros_like(scores)
        masked_scores = (
            masked_scores.scatter_(dim=1, index=topk_indices, src=topk_values)
            .transpose(0, 1)  # (E, T)
            .reshape(E, T, 1)
            .expand(E, T, D)
        )

        tokens = tokens.view(1, T, D).expand(E, T, D)
        masked_tokens = tokens * masked_scores

        # Routed Experts
        EG: int = self.EG
        if self.ep_size > 1:
            send_tokens = masked_tokens.contiguous()
            send_list = list(torch.chunk(send_tokens, chunks=self.ep_size, dim=0))
            recv_tokens = torch.empty_like(send_tokens)
            recv_list = list(torch.chunk(recv_tokens, chunks=self.ep_size, dim=0))

            torch.distributed.all_to_all(
                output_tensor_list=recv_list,
                input_tensor_list=send_list,
                group=self.ep_group,
            )

            masked_tokens = recv_tokens.reshape(EG, -1, D)

        routed_y = self._fake_quant(torch.bmm, masked_tokens, self.routed_experts.w13)
        routed_y0, routed_y1 = torch.chunk(routed_y, chunks=2, dim=-1)
        routed_z = routed_y0 * torch.sigmoid(routed_y0) * routed_y1
        routed_z = self._fake_quant(torch.bmm, routed_z, self.routed_experts.w2)

        if self.ep_size > 1:
            send_tokens = routed_z.reshape(E * T, D).contiguous()
            send_list = list(torch.chunk(send_tokens, chunks=self.ep_size, dim=0))
            recv_tokens = torch.empty_like(send_tokens)
            recv_list = list(torch.chunk(recv_tokens, chunks=self.ep_size, dim=0))

            torch.distributed.all_to_all(
                output_tensor_list=recv_list,
                input_tensor_list=send_list,
                group=self.ep_group,
            )

            routed_z = recv_tokens.reshape(E, T, D)

        return (shared_z + routed_z.sum(dim=0)).reshape(B, -1, D)

    def _fake_quant(self, op, x: torch.Tensor, w: ScaledParameter) -> torch.Tensor:
        if not w.is_scaled:
            return op(x, w.transpose(-1, -2))

        xq, xs = triton_quantize_fp8_row(x)
        wq, ws = w.weights, w.scales

        y = (
            op(xq.to(x.dtype), wq.transpose(-1, -2).to(x.dtype))
            * xs.unsqueeze(-1)
            * ws.unsqueeze(-2)
        )
        return y.to(x.dtype)


class MetaShufflingMoE(BaselineMoE):
    def __init__(
        self,
        ep_group: ProcessGroup,
        ep_mp_group: ProcessGroup,
        moe_args: MoEArgs,
    ) -> None:
        super().__init__(ep_group=ep_group, ep_mp_group=ep_mp_group, moe_args=moe_args)

        assert (
            self.mp_size == self.ep_mp_size
        ), "MetaShuffling only supports mp_size = mp_size_for_routed_experts now"

        assert (
            self.top_k == 1
        ), "MetaShuffling only supports top 1 routing at the moment"

        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.comp_end_event: torch.cuda.Event = torch.cuda.Event()
        self.comm_end_event: torch.cuda.Event = torch.cuda.Event()

        self.use_fast_accum: bool = moe_args.use_fast_accum
        self.dedup_comm: bool = moe_args.dedup_comm
        if self.dedup_comm:
            assert (
                self.ep_mp_size == self.mp_size
            ), "TP2EP is not supported for dedup at the moment."

        self.activation_scale_ub = None

    def forward(self, x: torch.Tensor, use_static_shape: bool) -> torch.Tensor:
        with torch.no_grad():
            if self.ep_size == 1:
                return self._no_comm_forward(x, use_static_shape)
            if use_static_shape:
                return self._static_comm_forward(x)
            else:
                return self._dynamic_comm_forward(x)

    def _dynamic_comm_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        comp_stream = torch.cuda.current_stream()

        (B, T, D) = tokens.shape
        T *= B

        # 1. Dispatch router kernels.
        routed_tokens, routed_tokens_scales, token_counts, token_indices = self._route(
            tokens
        )
        assert routed_tokens_scales is None

        # 2. Dispatch 1st all2all on shapes.
        self.comp_end_event.record()
        with torch.cuda.stream(self.comm_stream):
            self.comp_end_event.wait()

            send_token_counts = token_counts
            recv_token_counts = self._exchange_shapes(send_token_counts)
            send_token_counts.record_stream(self.comm_stream)

        recv_token_counts.record_stream(comp_stream)

        # 3. Dispatch shared expert part 1.
        shared_y = self._shared_expert_part1(tokens)

        with torch.cuda.stream(self.comm_stream):
            # 4. CPU/GPU sync.
            concat_counts = torch.concat(
                [send_token_counts.flatten(), recv_token_counts.flatten()]
            ).cpu()
            send_tokens_list = concat_counts[: self.E].tolist()
            recv_tokens_list = concat_counts[self.E :].tolist()

            # 5. Dispatch 2nd all2all on tokens.
            send_tokens = routed_tokens
            recv_tokens = self._exchange_tokens(
                send_tokens,
                send_tokens_list,
                recv_tokens_list,
                is_input=True,
            )
            send_tokens.record_stream(self.comm_stream)

            self.comm_end_event.record()
        recv_tokens.record_stream(comp_stream)

        # 6. Dispatch routed expert kernels.
        self.comm_end_event.wait()
        recv_T = recv_tokens.shape[0]
        assert recv_tokens.shape == (recv_T, D)
        assert recv_token_counts.shape == (self.ep_size, self.num_local_experts)
        shuffled_recv_tokens, shuffled_recv_token_counts = combine_shuffling(
            recv_tokens, recv_token_counts
        )
        assert shuffled_recv_tokens.shape == (recv_T, D)
        assert shuffled_recv_token_counts.shape == (self.num_local_experts + 1,)
        routed_z = self._routed_expert(
            shuffled_recv_tokens,
            shuffled_recv_token_counts[:-1],
        )
        assert routed_z.shape == (recv_T, D)
        shuffled_send_tokens = split_shuffling(routed_z, recv_token_counts)
        assert shuffled_send_tokens.shape == (recv_T, D)

        # 7. Dispatch 3rd all2all on tokens.
        self.comp_end_event.record()
        with torch.cuda.stream(self.comm_stream):
            self.comp_end_event.wait()

            send_tokens = shuffled_send_tokens
            recv_tokens = self._exchange_tokens(
                send_tokens,
                recv_tokens_list,
                send_tokens_list,
                is_input=False,
            )
            send_tokens.record_stream(self.comm_stream)

            self.comm_end_event.record()
        recv_tokens.record_stream(comp_stream)

        # 8. Dispatch shared expert part 2.
        shared_z = self._shared_expert_part2(shared_y)

        # 9. Dispatch combine outputs.
        self.comm_end_event.wait()
        final_output = self._combine_outputs(
            shared_z, recv_tokens, token_indices, token_counts, padded=False
        )

        T //= B
        return final_output.view(B, T, D)

    def _static_comm_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        comp_stream = torch.cuda.current_stream()

        (B, T, D) = tokens.shape
        T *= B

        # 1. Dispatch router kernels.
        routed_tokens, routed_tokens_scales, token_counts, token_indices = self._route(
            tokens
        )
        assert routed_tokens_scales is None

        # 2. Dispatch allgather on shapes and tokens.
        self.comp_end_event.record()
        with torch.cuda.stream(self.comm_stream):
            self.comp_end_event.wait()

            send_token_counts = token_counts
            send_tokens = routed_tokens
            # TODO(shikaili): Check if using 1 allgather is faster even with copies.
            recv_token_counts = self._gather_shapes(send_token_counts)
            recv_tokens = self._gather_tokens(send_tokens)
            send_token_counts.record_stream(self.comm_stream)
            send_tokens.record_stream(self.comm_stream)

            self.comm_end_event.record()
        recv_token_counts.record_stream(comp_stream)
        recv_tokens.record_stream(comp_stream)

        # 3. Dispatch shared expert part 1.
        shared_y = self._shared_expert_part1(tokens)

        # 4. Dispatch routed expert kernels.
        self.comm_end_event.wait()
        assert recv_tokens.shape == (
            self.ep_size,
            T,
            D,
        ), f"{recv_tokens.shape=}, {(self.ep_size, T, D)=}"
        assert recv_token_counts.shape == (self.ep_size, self.E)
        shuffled_recv_tokens, shuffled_recv_token_counts = combine_shuffling(
            recv_tokens.view(-1, D),
            recv_token_counts,
            expert_start=self.ep_rank * self.num_local_experts,
            expert_end=(self.ep_rank + 1) * self.num_local_experts,
        )
        assert shuffled_recv_tokens.shape == (self.ep_size * T, D)
        assert shuffled_recv_token_counts.shape == (
            self.num_local_experts + 1,
        ), f"{shuffled_recv_token_counts.shape=}"
        routed_z = self._routed_expert(
            shuffled_recv_tokens,
            shuffled_recv_token_counts[:-1],
        )
        assert routed_z.shape == (self.ep_size * T, D)
        shuffled_send_tokens = split_shuffling(
            routed_z,
            recv_token_counts,
            expert_start=self.ep_rank * self.num_local_experts,
            expert_end=(self.ep_rank + 1) * self.num_local_experts,
        )
        assert shuffled_send_tokens.shape == (self.ep_size * T, D)

        # 5. Dispatch all2all on tokens.
        self.comp_end_event.record()
        with torch.cuda.stream(self.comm_stream):
            self.comp_end_event.wait()

            send_tokens = shuffled_send_tokens
            recv_tokens = self._exchange_tokens(send_tokens, None, None, is_input=False)
            send_tokens.record_stream(self.comm_stream)

            self.comm_end_event.record()
        recv_tokens.record_stream(comp_stream)

        # 6. Dispatch shared expert part 2.
        shared_z = self._shared_expert_part2(shared_y)

        # 7. Dispatch combine outputs.
        self.comm_end_event.wait()
        final_output = self._combine_outputs(
            shared_z,
            recv_tokens.view(self.ep_size, T, D),
            token_indices,
            token_counts,
            padded=True,
        )

        T //= B
        return final_output.view(B, T, D)

    def _no_comm_forward(
        self, tokens: torch.Tensor, overlap_router_and_shared_expert: bool
    ) -> torch.Tensor:
        # Default stream for compute
        comp_stream = torch.cuda.current_stream()
        if overlap_router_and_shared_expert:
            self.comp_end_event.record()
        (B, T, D) = tokens.shape

        # 1. Dispatch router kernels and shared experts GEMMs.
        routed_tokens, routed_tokens_scales, token_counts, token_indices = self._route(
            tokens
        )

        if overlap_router_and_shared_expert:
            with torch.cuda.stream(self.comm_stream):
                self.comp_end_event.wait()

                shared_y = self._shared_expert_part1(tokens)
                shared_z = self._shared_expert_part2(shared_y)
                tokens.record_stream(self.comm_stream)

                self.comm_end_event.record()
            shared_z.record_stream(comp_stream)
            self.comm_end_event.wait()
        else:
            shared_y = self._shared_expert_part1(tokens)
            shared_z = self._shared_expert_part2(shared_y)

        # 2. Dispatch routed expert GEMMs.
        if not torch.version.hip:
            final_output = self._routed_expert(
                routed_tokens,
                token_counts,
                token_scales=routed_tokens_scales,
                shared_output=shared_z,
                token_indices=token_indices,
            )
        else:
            routed_z = self._routed_expert(
                routed_tokens,
                token_counts,
                token_scales=routed_tokens_scales,
            )
            # 3. Dispatch combine outputs.
            final_output = self._combine_outputs(
                shared_z, routed_z, token_indices, token_counts, padded=False
            )

        return final_output.view(B, T, D)

    def _exchange_shapes(self, send_sizes: torch.Tensor) -> torch.Tensor:
        "No CPU/GPU sync in this function."
        if self.ep_size == 1:
            return send_sizes

        assert tuple(send_sizes.shape) == (self.E,)
        recv_sizes = torch.empty_like(send_sizes)

        recv_sizes_list = list(recv_sizes.chunk(self.ep_size))
        send_sizes_list = list(send_sizes.chunk(self.ep_size))

        assert all(r.is_contiguous() for r in recv_sizes_list)
        assert all(s.is_contiguous() for s in send_sizes_list)
        torch.distributed.all_to_all(
            output_tensor_list=recv_sizes_list,
            input_tensor_list=send_sizes_list,
            group=self.ep_group,
        )

        # send_sizes: [E] viewed as [EP, EG]
        # recv_sizes: [E] viewed as [EP, EG]
        return recv_sizes.view(self.ep_size, self.num_local_experts)

    def _gather_shapes(self, send_sizes: torch.Tensor) -> torch.Tensor:
        "No CPU/GPU sync in this function."
        if self.ep_size == 1:
            return send_sizes

        assert tuple(send_sizes.shape) == (self.E,)
        recv_sizes = torch.empty(
            (self.ep_size, self.E), dtype=send_sizes.dtype, device=send_sizes.device
        )

        assert send_sizes.is_contiguous()
        assert recv_sizes.is_contiguous()
        torch.distributed.all_gather_into_tensor(
            output_tensor=recv_sizes,
            input_tensor=send_sizes,
            group=self.ep_group,
        )

        # send_sizes: [E]
        # recv_sizes: [EP, E]
        return recv_sizes

    def _exchange_tokens(
        self,
        send_tokens: torch.Tensor,
        send_sizes: Optional[list[int]],
        recv_sizes: Optional[list[int]],
        is_input: bool,
    ) -> torch.Tensor:
        """
        When `send_sizes`/`recv_size` are `None`, we assume the tokens are evenly distributed
        across different EP ranks, so the total number of tokens `T` are split by `E`.
        No CPU/GPU sync in this function.
        """
        if self.ep_size == 1:
            return send_tokens

        D = send_tokens.shape[-1]
        send_tokens = send_tokens.view(-1, D)
        T = send_tokens.shape[0]

        if send_sizes is None:
            send_sizes = [T // self.ep_size for _ in range(self.ep_size)]
        else:
            send_sizes = [
                sum(
                    send_sizes[
                        r * self.num_local_experts : (r + 1) * self.num_local_experts
                    ]
                )
                for r in range(self.ep_size)
            ]

        if recv_sizes is None:
            recv_sizes = [T // self.ep_size for _ in range(self.ep_size)]
        else:
            recv_sizes = [
                sum(
                    recv_sizes[
                        r * self.num_local_experts : (r + 1) * self.num_local_experts
                    ]
                )
                for r in range(self.ep_size)
            ]

        # TODO: Add FP8 A2A to example.
        if self.dedup_comm:
            if is_input:
                sliced_recv_tokens = torch.empty(
                    (sum(none_throws(recv_sizes)), D // self.ep_mp_size),
                    dtype=send_tokens.dtype,
                    device=send_tokens.device,
                )
                # TODO(shikaili): Extremely high copy overhead in prefill.
                sliced_send_tokens = send_tokens.chunk(self.ep_mp_size, dim=-1)[
                    self.ep_mp_rank
                ].contiguous()

                recv_tokens_list = list(
                    sliced_recv_tokens.split(none_throws(recv_sizes))
                )
                send_tokens_list = list(
                    sliced_send_tokens.split(none_throws(send_sizes))
                )

                assert all(r.is_contiguous() for r in recv_tokens_list)
                assert all(s.is_contiguous() for s in send_tokens_list)
                torch.distributed.all_to_all(
                    output_tensor_list=recv_tokens_list,
                    input_tensor_list=send_tokens_list,
                    group=self.ep_group,
                )

                recv_tokens_permutated = torch.empty(
                    (
                        self.ep_mp_size,
                        sum(none_throws(recv_sizes)),
                        D // self.ep_mp_size,
                    ),
                    dtype=send_tokens.dtype,
                    device=send_tokens.device,
                )

                assert sliced_recv_tokens.is_contiguous()
                assert recv_tokens_permutated.is_contiguous()
                torch.distributed.all_gather_into_tensor(
                    output_tensor=recv_tokens_permutated,
                    input_tensor=sliced_recv_tokens,
                    group=self.ep_mp_group,
                )

                return (
                    recv_tokens_permutated.permute(1, 0, 2).reshape(-1, D).contiguous()
                )
            else:
                # ReduceScatter
                reduced_sliced_send_tokens = torch.empty(
                    (D // self.ep_mp_size, sum(none_throws(send_sizes))),
                    dtype=send_tokens.dtype,
                    device=send_tokens.device,
                )
                torch.distributed.reduce_scatter_tensor(
                    output=reduced_sliced_send_tokens,
                    input=send_tokens.transpose(0, 1).contiguous(),
                    group=self.ep_mp_group,
                )
                reduced_sliced_send_tokens = reduced_sliced_send_tokens.transpose(
                    0, 1
                ).contiguous()

                # AlltoAll
                reduced_sliced_recv_tokens = torch.empty(
                    (sum(none_throws(recv_sizes)), D // self.ep_mp_size),
                    dtype=send_tokens.dtype,
                    device=send_tokens.device,
                )
                recv_tokens_list = list(
                    reduced_sliced_recv_tokens.split(none_throws(recv_sizes))
                )
                send_tokens_list = list(
                    reduced_sliced_send_tokens.split(none_throws(send_sizes))
                )

                assert all(r.is_contiguous() for r in recv_tokens_list)
                assert all(s.is_contiguous() for s in send_tokens_list)
                torch.distributed.all_to_all(
                    output_tensor_list=recv_tokens_list,
                    input_tensor_list=send_tokens_list,
                    group=self.ep_group,
                )

                # Padding
                slice_d = D // self.ep_mp_size
                pad_l = slice_d * self.ep_mp_rank
                pad_r = D - pad_l - slice_d
                return torch.nn.functional.pad(
                    reduced_sliced_recv_tokens, (pad_l, pad_r)
                )
        else:
            recv_tokens = torch.empty(
                (sum(none_throws(recv_sizes)), D),
                dtype=send_tokens.dtype,
                device=send_tokens.device,
            )

            recv_tokens_list = list(recv_tokens.split(none_throws(recv_sizes)))
            send_tokens_list = list(send_tokens.split(none_throws(send_sizes)))

            assert all(r.is_contiguous() for r in recv_tokens_list)
            assert all(s.is_contiguous() for s in send_tokens_list)
            torch.distributed.all_to_all(
                output_tensor_list=recv_tokens_list,
                input_tensor_list=send_tokens_list,
                group=self.ep_group,
            )

        return recv_tokens

    def _gather_tokens(
        self,
        send_tokens: torch.Tensor,
    ) -> torch.Tensor:
        "No CPU/GPU sync in this function."
        if self.ep_size == 1:
            return send_tokens

        # TODO: Add FP8 AG to example.
        T, D = send_tokens.shape
        if self.dedup_comm:
            inter_node_recv_tokens = torch.empty(
                (self.ep_size, T, D // self.ep_mp_size),
                dtype=send_tokens.dtype,
                device=send_tokens.device,
            )
            # Copy overhead.
            inter_node_send_tokens = send_tokens.chunk(self.ep_mp_size, dim=-1)[
                self.ep_mp_rank
            ].contiguous()

            assert inter_node_send_tokens.is_contiguous()
            assert inter_node_recv_tokens.is_contiguous()
            torch.distributed.all_gather_into_tensor(
                output_tensor=inter_node_recv_tokens,
                input_tensor=inter_node_send_tokens,
                group=self.ep_group,
            )

            intra_node_recv_tokens_transposed = torch.empty(
                (self.ep_mp_size, self.ep_size, T, D // self.ep_mp_size),
                dtype=send_tokens.dtype,
                device=send_tokens.device,
            )

            assert inter_node_recv_tokens.is_contiguous()
            assert intra_node_recv_tokens_transposed.is_contiguous()
            torch.distributed.all_gather_into_tensor(
                output_tensor=intra_node_recv_tokens_transposed,
                input_tensor=inter_node_recv_tokens,
                group=self.ep_mp_group,
            )

            # Copy overhead.
            return (
                intra_node_recv_tokens_transposed.permute(1, 2, 0, 3)
                .reshape(self.ep_size, T, D)
                .contiguous()
            )
        else:
            recv_tokens = torch.empty(
                (self.ep_size, T, D),
                dtype=send_tokens.dtype,
                device=send_tokens.device,
            )

            assert send_tokens.is_contiguous()
            assert recv_tokens.is_contiguous()
            torch.distributed.all_gather_into_tensor(
                output_tensor=recv_tokens,
                input_tensor=send_tokens,
                group=self.ep_group,
            )
            return recv_tokens

    def _route(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        B, T, D = tokens.shape
        tokens = tokens.view(-1, D)

        assert not self.router_DE.is_scaled
        scores = torch.nn.functional.linear(tokens, self.router_DE.T)
        scores = torch.sigmoid(scores)
        assert scores.shape == (B * T, self.E)

        token_counts, expert_indices, token_indices = index_shuffling(
            scores,  # num_tokens
        )
        token_counts = token_counts[: self.E]

        if self.dedup_comm:
            split_sizes = [
                token_counts.shape[0],
                expert_indices.shape[0],
                token_indices.shape[0],
            ]
            output = torch.concat([token_counts, expert_indices, token_indices], dim=0)
            # Require broadcast as index_shuffling is not deterministic.
            torch.distributed.broadcast(
                output,
                src=(torch.distributed.get_rank() // self.ep_mp_size) * self.ep_mp_size,
                group=self.ep_mp_group,
            )
            token_counts, expert_indices, token_indices = torch.split(
                output, split_sizes, dim=0
            )

        if self.is_routed_fp8_rowwise and self.ep_size == 1:
            routed_tokens, routed_tokens_scales = gather_scale_quant_dense_tokens(
                tokens,
                token_indices=token_indices.flatten(),
                expert_indices=expert_indices.flatten(),
                scores=scores,
                scale_ub=self.activation_scale_ub,
            )
        else:
            routed_tokens = gather_scale_dense_tokens(
                tokens,
                token_indices=token_indices.flatten(),
                expert_indices=expert_indices.flatten(),
                scores=scores,
            )
            routed_tokens_scales = None
        return routed_tokens, routed_tokens_scales, token_counts, token_indices

    def _shared_expert_part1(self, x: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, D]
        D = x.shape[-1]
        x = x.view(-1, D)
        w13 = self.shared_experts.w13

        if not self.is_shared_fp8_rowwise:
            # TODO(shikaili): Skip padded tokens.
            return x @ w13.T
        else:
            x, x_scale = triton_quantize_fp8_row(x, self.activation_scale_ub)
            # TODO(shikaili): Skip padded tokens.
            return torch.ops.fbgemm.f8f8bf16_rowwise(
                x,
                w13.weights,
                x_scale,
                w13.scales,
                use_fast_accum=self.use_fast_accum,
            )

    def _shared_expert_part2(self, y: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, D]
        HD_L_2 = y.shape[-1]
        HD_L = HD_L_2 // 2
        w2 = self.shared_experts.w2

        z, z_scale = self._fused_silu_mul(
            y[:, :HD_L],
            y[:, HD_L:],
            self.is_shared_fp8_rowwise,
            self.activation_scale_ub,
        )
        if not self.is_shared_fp8_rowwise:
            assert z_scale is None
            # TODO(shikaili): Skip padded tokens.
            return z @ w2.T
        else:
            assert z_scale is not None
            # TODO(shikaili): Skip padded tokens.
            return torch.ops.fbgemm.f8f8bf16_rowwise(
                z,
                w2.weights,
                z_scale,
                w2.scales,
                use_fast_accum=self.use_fast_accum,
            )

    def _routed_expert(
        self,
        tokens: torch.Tensor,
        token_counts: torch.Tensor,
        token_scales: Optional[torch.Tensor] = None,
        shared_output: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # tokens: [B, T, D]
        D = tokens.shape[-1]
        x = tokens.view(-1, D)

        if x.shape[0] == 0:
            return x

        w13 = self.routed_experts.w13
        w2 = self.routed_experts.w2

        assert D == w13.shape[-1]
        HD_L = w2.shape[-1]

        assert token_counts.shape == (self.num_local_experts,)
        if not self.is_routed_fp8_rowwise:
            y = grouped_gemm(
                x,
                w13.view(-1, D),
                token_counts,
                use_fast_accum=self.use_fast_accum,
                _use_warp_specialization=not torch.version.hip,
            )
            z, _ = self._fused_silu_mul(y[:, :HD_L], y[:, HD_L:], False)
            return grouped_gemm(
                z,
                w2.view(-1, HD_L),
                token_counts,
                use_fast_accum=self.use_fast_accum,
                _use_warp_specialization=not torch.version.hip,
                _output_tensor=shared_output,
                _scatter_add_indices=token_indices,
            )
        else:
            if token_scales is None:
                x, x_scale = triton_quantize_fp8_row(x, self.activation_scale_ub)
            else:
                x_scale = token_scales
            y = grouped_gemm_fp8_rowwise(
                x,
                w13.weights.view(-1, D),
                token_counts,
                x_scale.view(-1),
                w13.scales.view(-1),
                use_fast_accum=self.use_fast_accum,
                _use_warp_specialization=not torch.version.hip,
            )
            # TODO(shikaili): Skip padded tokens.
            z, z_scale = self._fused_silu_mul(
                y[:, :HD_L], y[:, HD_L:], True, self.activation_scale_ub
            )
            assert z_scale is not None
            return grouped_gemm_fp8_rowwise(
                z,
                w2.weights.view(-1, HD_L),
                token_counts,
                z_scale.view(-1),
                w2.scales.view(-1),
                use_fast_accum=self.use_fast_accum,
                _use_warp_specialization=not torch.version.hip,
                _output_tensor=shared_output,
                _scatter_add_indices=token_indices,
            )

    def _combine_outputs(
        self,
        shared_output_tokens: torch.Tensor,
        routed_output_tokens: torch.Tensor,
        token_indices: torch.Tensor,
        token_counts: torch.Tensor,
        padded: bool = False,
    ) -> torch.Tensor:
        D = shared_output_tokens.shape[-1]
        assert routed_output_tokens.shape[-1] == D

        if padded:
            scatter_add_padded_tokens(
                in_tokens=routed_output_tokens,
                token_counts=token_counts,
                token_indices=token_indices,
                out_tokens=shared_output_tokens,
            )
            return shared_output_tokens

        scatter_add_dense_tokens(
            shared_output_tokens,
            routed_output_tokens.view(-1, D),
            token_indices,
        )
        return shared_output_tokens

    def _fused_silu_mul(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        is_fp8: bool,
        scale_ub: Optional[torch.Tensor] = None,
    ):
        z_scale = None
        if is_fp8:
            z, z_scale = silu_mul_quant(x0, x1, scale_ub)
        else:
            z = silu_mul(x0, x1)
        return z, z_scale
