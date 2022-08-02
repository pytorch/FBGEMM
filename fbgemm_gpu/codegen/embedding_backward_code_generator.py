#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jinja2

args: argparse.Namespace
_: List[str]
TENSOR: int
INT_TENSOR: int
LONG_TENSOR: int
INT: int
FLOAT: int


parser = argparse.ArgumentParser()
# By default the source template files are in the same folder as
# embedding_backward_code_generator.py;
# The install dir is by default the same as the current folder.
parser.add_argument("--install_dir", default=".", help="where to put generated file")
parser.add_argument("--opensource", action="store_false", dest="is_fbcode")
args, _ = parser.parse_known_args()


env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
)
# Upper Limit of "max_embedding_dim (max_D)":
# BT_block_size * sizeof(float) * 4 * kWarpSize * {{ kMaxVecsPerThread }}
# needs to be smaller than the allocated shared memory size (2/3 of 96 KB
# on V100 and 160 KB on A100.
# BT_block_size * 4 * 4 * 32 * (max_D // 128) <= 64 * 1024 (V100) or 96 * 1024 (A100)
# Since BT_block_size >= 1, max_D <= 16K (V100) or 24K (A100).
# Note that if we increase max_D, it will increase the compilation time significantly.
env.globals["max_embedding_dim"] = 1024
env.globals["dense"] = False


def write(filename: str, s: str) -> None:
    with open(os.path.join(args.install_dir, filename), "w") as f:
        f.write(s)


def _arg_constructor(
    type: str, name: str, gpu: bool = True, precision: int = 32
) -> str:
    return (
        f"{name}.packed_accessor{precision}<{type}, 1, at::RestrictPtrTraits>()"
        if gpu
        else f"{name}.accessor<{type}, 1>()"
    )


def _arg(type: str, name: str, gpu: bool = True, precision: int = 32) -> str:
    return (
        f"at::PackedTensorAccessor{precision}<{type}, 1, at::RestrictPtrTraits> {name}"
        if gpu
        else f"at::TensorAccessor<{type}, 1> {name}"
    )


def acc_cache_tensor_arg_constructor(name: str, gpu: bool = True) -> str:
    return _arg_constructor(
        "at::acc_type<" + ("cache_t" if gpu else "scalar_t") + ", true>",
        name,
        gpu=gpu,
        precision=64,
    )


def acc_cache_tensor_arg(name: str, gpu: bool = True) -> str:
    return _arg(
        "at::acc_type<" + ("cache_t" if gpu else "scalar_t") + ", true>",
        name,
        gpu=gpu,
        precision=64,
    )


def long_tensor_arg_constructor(name: str, gpu: bool = True) -> str:
    return _arg_constructor("int64_t", name, gpu=gpu)


def long_tensor_arg(name: str, gpu: bool = True) -> str:
    return _arg("int64_t", name, gpu=gpu)


def int_tensor_arg_constructor(name: str, gpu: bool = True) -> str:
    return _arg_constructor("int32_t", name, gpu=gpu)


def int_tensor_arg(name: str, gpu: bool = True) -> str:
    return _arg("int32_t", name, gpu=gpu)


def tensor_arg(name: str) -> str:
    return f"Tensor {name}"


def double_arg(name: str, default: float = 0.0) -> str:
    return f"double {name} = {default}"


def float_arg(name: str, default: float = 0.0) -> str:
    return f"float {name} = {default}"


def int64_arg(name: str, default: int = 0) -> str:
    return f"int64_t {name} = {default}"


def int_arg(name: str, default: int = 0) -> str:
    return f"int {name} = {default}"


def generate(**kwargs: Any) -> None:
    gen_args = kwargs["args"]

    # Generates CUDA variants.
    kwargs["args"] = gen_args["cuda"]

    template = env.get_template("embedding_backward_split_template.cu")
    src_cu = template.render(weighted=False, **kwargs)
    write(
        f"gen_embedding_backward_{kwargs.get('optimizer')}_split_unweighted_cuda.cu",
        src_cu,
    )
    src_cu = template.render(weighted=True, **kwargs)
    write(
        f"gen_embedding_backward_{kwargs.get('optimizer')}_split_weighted_cuda.cu",
        src_cu,
    )
    if not kwargs.get("dense"):
        template = env.get_template("embedding_backward_split_host_template.cpp")
        src_cpp = template.render(**kwargs)
        write(f"gen_embedding_backward_split_{kwargs.get('optimizer')}.cpp", src_cpp)

        # Generates Python invoker for CUDA + CPU
        template = env.get_template("split_embedding_codegen_lookup_invoker.template")
        src_py = template.render(is_fbcode=args.is_fbcode, **kwargs)
        write(f"lookup_{kwargs.get('optimizer')}.py", src_py)

    # Generates CPU variants.
    kwargs["args"] = gen_args["cpu"]

    is_approx = "approx" in kwargs.get("optimizer")
    template = (
        env.get_template("embedding_backward_split_cpu_approx_template.cpp")
        if is_approx
        else env.get_template("embedding_backward_split_cpu_template.cpp")
    )

    src_cpp = template.render(**kwargs)
    write(
        f"gen_embedding_backward_{kwargs.get('optimizer')}_split_cpu.cpp",
        src_cpp,
    )

    if not kwargs.get("dense"):
        template = env.get_template("embedding_backward_split_host_cpu_template.cpp")
        src_cpp = template.render(**kwargs)
        write(
            f"gen_embedding_backward_split_{kwargs.get('optimizer')}_cpu.cpp", src_cpp
        )


@dataclass
class Args:
    split_kernel_args: List[str]
    split_kernel_arg_constructors: List[str]
    split_cpu_kernel_args: List[str]
    split_cpu_kernel_arg_constructors: List[str]
    split_function_args: List[str]
    split_saved_tensors: List[str]
    split_tensors: List[str]
    saved_data: List[Tuple[str, str]]
    split_function_arg_names: List[str]
    split_function_schemas: List[str]
    split_variables: List[str]


TENSOR, INT_TENSOR, LONG_TENSOR, INT, FLOAT = range(5)


def make_args(
    arg_spec: List[Union[Tuple[int, str], Tuple[int, str, Union[float, int]]]]
) -> Dict[str, Any]:
    def make_kernel_arg(ty: int, name: str, default: Union[int, float]) -> str:
        return {
            TENSOR: acc_cache_tensor_arg,
            INT_TENSOR: int_tensor_arg,
            LONG_TENSOR: long_tensor_arg,
            INT: lambda x: int64_arg(x, default=int(default)),
            FLOAT: lambda x: float_arg(x, default=default),
        }[ty](name)

    def make_kernel_arg_constructor(ty: int, name: str) -> str:
        return {
            TENSOR: acc_cache_tensor_arg_constructor,
            INT_TENSOR: int_tensor_arg_constructor,
            LONG_TENSOR: long_tensor_arg_constructor,
            INT: lambda x: x,
            FLOAT: lambda x: x,
        }[ty](name)

    def make_cpu_kernel_arg(ty: int, name: str, default: Union[int, float]) -> str:
        return {
            TENSOR: lambda x: acc_cache_tensor_arg(x, gpu=False),
            INT_TENSOR: lambda x: int_tensor_arg(x, gpu=False),
            LONG_TENSOR: lambda x: long_tensor_arg(x, gpu=False),
            INT: lambda x: int64_arg(x, default=int(default)),
            FLOAT: lambda x: float_arg(x, default=default),
        }[ty](name)

    def make_cpu_kernel_arg_constructor(ty: int, name: str) -> str:
        return {
            TENSOR: lambda x: acc_cache_tensor_arg_constructor(x, gpu=False),
            INT_TENSOR: lambda x: int_tensor_arg_constructor(x, gpu=False),
            LONG_TENSOR: lambda x: long_tensor_arg_constructor(x, gpu=False),
            INT: lambda x: x,
            FLOAT: lambda x: x,
        }[ty](name)

    def make_function_arg(ty: int, name: str, default: Union[int, float]) -> str:
        return {
            TENSOR: tensor_arg,
            INT_TENSOR: tensor_arg,
            LONG_TENSOR: tensor_arg,
            INT: lambda x: int64_arg(x, default=int(default)),
            FLOAT: lambda x: double_arg(x, default=default),
        }[ty](name)

    def make_function_schema_arg(ty: int, name: str, default: Union[int, float]) -> str:
        return {
            TENSOR: tensor_arg,
            INT_TENSOR: tensor_arg,
            LONG_TENSOR: tensor_arg,
            INT: lambda x: int_arg(x, default=int(default)),
            FLOAT: lambda x: float_arg(x, default=default),
        }[ty](name)

    def make_ivalue_cast(ty: int) -> str:
        return {INT: "toInt", FLOAT: "toDouble"}[ty]

    def make_args_for_compute_device(
        split_arg_spec: List[Tuple[int, str, Union[int, float]]]
    ) -> Args:
        return Args(
            split_kernel_args=[
                make_kernel_arg(ty, name, default)
                for (ty, name, default) in split_arg_spec
            ],
            split_kernel_arg_constructors=[
                make_kernel_arg_constructor(ty, name)
                for (ty, name, default) in split_arg_spec
            ],
            split_cpu_kernel_args=[
                make_cpu_kernel_arg(ty, name, default)
                for (ty, name, default) in split_arg_spec
            ],
            split_cpu_kernel_arg_constructors=[
                make_cpu_kernel_arg_constructor(ty, name)
                for (ty, name, default) in split_arg_spec
            ],
            split_function_args=[
                make_function_arg(ty, name, default)
                for (ty, name, default) in split_arg_spec
            ],
            split_tensors=[
                name for (ty, name, default) in augmented_arg_spec if ty == TENSOR
            ],
            split_saved_tensors=[
                name
                for (ty, name, default) in split_arg_spec
                if ty in (TENSOR, INT_TENSOR, LONG_TENSOR)
            ],
            saved_data=[
                (name, make_ivalue_cast(ty))
                for (ty, name, default) in augmented_arg_spec
                if ty != TENSOR
            ],
            split_function_arg_names=[name for (ty, name, default) in split_arg_spec],
            split_function_schemas=[
                make_function_schema_arg(ty, name, default)
                for (ty, name, default) in split_arg_spec
            ],
            split_variables=["Variable()" for _ in split_arg_spec],
        )

    DEFAULT_ARG_VAL = 0
    augmented_arg_spec = [
        item if len(item) == 3 else (*item, DEFAULT_ARG_VAL) for item in arg_spec
    ]

    split_arg_spec = []
    for (ty, arg, default) in augmented_arg_spec:
        if ty in (FLOAT, INT):
            split_arg_spec.append((ty, arg, default))
        else:
            assert ty == TENSOR
            split_arg_spec.extend(
                [
                    (TENSOR, f"{arg}_host", default),
                    (INT_TENSOR, f"{arg}_placements", default),
                    (LONG_TENSOR, f"{arg}_offsets", default),
                ]
            )
    cpu = make_args_for_compute_device(split_arg_spec)

    split_arg_spec = []
    for (ty, arg, default) in augmented_arg_spec:
        if ty in (FLOAT, INT):
            split_arg_spec.append((ty, arg, default))
        else:
            assert ty == TENSOR
            split_arg_spec.extend(
                [
                    (TENSOR, f"{arg}_dev", default),
                    (TENSOR, f"{arg}_uvm", default),
                    (INT_TENSOR, f"{arg}_placements", default),
                    (LONG_TENSOR, f"{arg}_offsets", default),
                ]
            )
    cuda = make_args_for_compute_device(split_arg_spec)

    return {"cpu": cpu, "cuda": cuda}


def adagrad() -> None:
    split_weight_update = """
      Vec4T<cache_t> m_t(&momentum1[idx * D + d]);
      m_t.acc.x += grad.acc.x * grad.acc.x;
      m_t.acc.y += grad.acc.y * grad.acc.y;
      m_t.acc.z += grad.acc.z * grad.acc.z;
      m_t.acc.w += grad.acc.w * grad.acc.w;
      m_t.store(&momentum1[idx * D + d]);

      weight_new.acc.x -= learning_rate * grad.acc.x / (sqrtf(m_t.acc.x) + eps);
      weight_new.acc.y -= learning_rate * grad.acc.y / (sqrtf(m_t.acc.y) + eps);
      weight_new.acc.z -= learning_rate * grad.acc.z / (sqrtf(m_t.acc.z) + eps);
      weight_new.acc.w -= learning_rate * grad.acc.w / (sqrtf(m_t.acc.w) + eps);
    """
    split_weight_update_cpu = """
      for (int64_t d = 0; d < D; ++d) {
        momentum1_host[embedding_begin + d] +=
            grad_buffer[d] * grad_buffer[d];
        host_weights_data[embedding_begin + d] -=
            learning_rate * grad_buffer[d] /
            (sqrt(momentum1_host[embedding_begin + d]) + eps);
      }
    """

    generate(
        optimizer="adagrad",
        args=make_args(
            [(TENSOR, "momentum1"), (FLOAT, "eps"), (FLOAT, "learning_rate")]
        ),
        split_precomputation="",
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def table_info_precomputation(momentum_prefix: str = "momentum1") -> str:
    template = """
      // table_begin -> (E, D, {momentum_prefix}_row_begin).
      std::map<int64_t, std::tuple<int64_t, int64_t, int64_t>> table_info_map;
      for (int64_t t = 0; t < T; ++t) {
        const auto D = D_offsets_data[t + 1] - D_offsets_data[t];
        const auto table_begin = weights_offsets_data[t];
        const auto {momentum_prefix}_row_begin = {momentum_prefix}_offsets_data[t];
        table_info_map[table_begin] = std::make_tuple(0, D, {momentum_prefix}_row_begin);
      }
      int64_t previous_table_begin = host_weights.numel();
      // NOTE: table_info_map is sorted by table_begin!
      for (auto it = table_info_map.rbegin(); it != table_info_map.rend(); ++it) {
        const auto D = std::get<1>(it->second);
        // Calculates number of rows of each table.
        std::get<0>(it->second) = (previous_table_begin - it->first) / D;
        previous_table_begin = it->first;
      }
    """
    return template.replace("{momentum_prefix}", momentum_prefix)


def rowwise_adagrad() -> None:
    split_weight_update = """
        weight_new.acc.x = correction * weight_new.acc.x - multiplier * grad.acc.x;
        weight_new.acc.y = correction * weight_new.acc.y - multiplier * grad.acc.y;
        weight_new.acc.z = correction * weight_new.acc.z - multiplier * grad.acc.z;
        weight_new.acc.w = correction * weight_new.acc.w - multiplier * grad.acc.w;
    """
    split_precomputation = """
    at::acc_type<cache_t, true> g_local_sum_square = 0.0;
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        auto gx = grad_sum[i].acc.x;
        auto gy = grad_sum[i].acc.y;
        auto gz = grad_sum[i].acc.z;
        auto gw = grad_sum[i].acc.w;
        if (weight_decay_mode == 1) {
            // L2 regularization
            int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
            Vec4T<at::acc_type<cache_t, true>> weight = weight_row_template.load(d, qparams_template);
            gx += weight_decay * weight.acc.x;
            gy += weight_decay * weight.acc.y;
            gz += weight_decay * weight.acc.z;
            gw += weight_decay * weight.acc.w;
        }
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
    }
    const at::acc_type<cache_t, true> g_avg_square =
        warpReduceAllSum<at::acc_type<cache_t, true>>(g_local_sum_square) / D;

    at::acc_type<cache_t, true> multiplier;
    at::acc_type<cache_t, true> correction;
    if (threadIdx.x == 0) {
        at::acc_type<cache_t, true> new_sum_square_grads = momentum1[idx] + g_avg_square;
        momentum1[idx] = new_sum_square_grads;
        multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);
        if (weight_decay_mode == 1) {
            // L2 regularization
            correction = 1.0 - multiplier * weight_decay;
        } else if (weight_decay_mode == 2) {
            // Decoupled weight decay
            correction = 1.0 - learning_rate * weight_decay;
        } else {
            // default value
            correction = 1.0;
        }
    }
    multiplier = shfl_sync(multiplier, 0);
    correction = shfl_sync(correction, 0);
    """
    split_weight_update_cpu = """
        at::acc_type<grad_t, true> g_local_sum_square = 0.0;
        for (int64_t d = 0; d < D; ++d) {
            auto grad = grad_buffer[d];
            if (weight_decay_mode == 1) {
                // L2 regularization
                grad += weight_decay * host_weights_data[embedding_begin + d];
            }
            g_local_sum_square += grad * grad;
        }
        auto g_avg_square = g_local_sum_square / D;
        at::acc_type<grad_t, true> new_sum_square_grads = momentum1_host[momentum1_offsets_data[feature_begin] + idx] + g_avg_square;
        momentum1_host[momentum1_offsets_data[feature_begin] + idx] = new_sum_square_grads;
        at::acc_type<grad_t, true> multiplier;
        multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);
        at::acc_type<grad_t, true> correction;
        if (weight_decay_mode == 1) {
            // L2 regularization
            correction = 1.0 - multiplier * weight_decay;
        } else if (weight_decay_mode == 2) {
            // Decoupled weight decay
            correction = 1.0 - learning_rate * weight_decay;
        } else {
            // default value
            correction = 1.0;
        }
        for (int64_t d = 0; d < D; ++d) {
            host_weights_data[embedding_begin + d] = correction * host_weights_data[embedding_begin + d] - grad_buffer[d] * multiplier;
        }
    """

    generate(
        optimizer="rowwise_adagrad",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (FLOAT, "eps"),
                (FLOAT, "learning_rate"),
                (FLOAT, "weight_decay", 0.0),
                (INT, "weight_decay_mode", 1),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )

    approx_split_weight_update = """
      // dummy computation to avoid unused variable warning
      weight_new.fma_(grad, -multiplier);
      assert(false); // approx rowwise AdaGrad is not supported on GPU
    """

    generate(
        optimizer="approx_rowwise_adagrad",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (FLOAT, "eps"),
                (FLOAT, "learning_rate"),
                (FLOAT, "weight_decay", 0.0),
                (INT, "weight_decay_mode", 1),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=approx_split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def rowwise_adagrad_with_weight_decay() -> None:
    split_weight_update = """
        weight_new.acc.x = correction * weight_new.acc.x - multiplier * grad.acc.x;
        weight_new.acc.y = correction * weight_new.acc.y - multiplier * grad.acc.y;
        weight_new.acc.z = correction * weight_new.acc.z - multiplier * grad.acc.z;
        weight_new.acc.w = correction * weight_new.acc.w - multiplier * grad.acc.w;
    """
    split_precomputation = """
    at::acc_type<cache_t, true> g_local_sum_square = 0.0;
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        auto gx = grad_sum[i].acc.x;
        auto gy = grad_sum[i].acc.y;
        auto gz = grad_sum[i].acc.z;
        auto gw = grad_sum[i].acc.w;
        if (weight_decay_mode == 1) {
            // L2 regularization
            int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
            Vec4T<at::acc_type<cache_t, true>> weight = weight_row_template.load(d, qparams_template);
            gx += weight_decay * weight.acc.x;
            gy += weight_decay * weight.acc.y;
            gz += weight_decay * weight.acc.z;
            gw += weight_decay * weight.acc.w;
        }
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
    }
    const at::acc_type<cache_t, true> g_avg_square =
        warpReduceAllSum<at::acc_type<cache_t, true>>(g_local_sum_square) / D;

    at::acc_type<cache_t, true> multiplier;
    at::acc_type<cache_t, true> correction;
    if (threadIdx.x == 0) {
        at::acc_type<cache_t, true> new_sum_square_grads = momentum1[idx] + g_avg_square;
        momentum1[idx] = new_sum_square_grads;
        multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);
        if (weight_decay_mode == 1) {
            // L2 regularization
            correction = 1.0 - multiplier * weight_decay;
        } else if (weight_decay_mode == 2) {
            // Decoupled weight decay
            correction = 1.0 - learning_rate * weight_decay;
        } else {
            // default value
            correction = 1.0;
        }
    }
    multiplier = shfl_sync(multiplier, 0);
    correction = shfl_sync(correction, 0);
    """
    split_weight_update_cpu = """
        at::acc_type<grad_t, true> g_local_sum_square = 0.0;
        for (int64_t d = 0; d < D; ++d) {
            auto grad = grad_buffer[d];
            if (weight_decay_mode == 1) {
                // L2 regularization
                grad += weight_decay * host_weights_data[embedding_begin + d];
            }
            g_local_sum_square += grad * grad;
        }
        auto g_avg_square = g_local_sum_square / D;
        at::acc_type<grad_t, true> new_sum_square_grads = momentum1_host[momentum1_offsets_data[feature_begin] + idx] + g_avg_square;
        momentum1_host[momentum1_offsets_data[feature_begin] + idx] = new_sum_square_grads;
        at::acc_type<grad_t, true> multiplier;
        multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);
        at::acc_type<grad_t, true> correction;
        if (weight_decay_mode == 1) {
            // L2 regularization
            correction = 1.0 - multiplier * weight_decay;
        } else if (weight_decay_mode == 2) {
            // Decoupled weight decay
            correction = 1.0 - learning_rate * weight_decay;
        } else {
            // default value
            correction = 1.0;
        }
        for (int64_t d = 0; d < D; ++d) {
            host_weights_data[embedding_begin + d] = correction * host_weights_data[embedding_begin + d] - grad_buffer[d] * multiplier;
        }
    """

    generate(
        optimizer="rowwise_adagrad_with_weight_decay",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (FLOAT, "eps"),
                (FLOAT, "learning_rate"),
                (FLOAT, "weight_decay", 0.0),
                (INT, "weight_decay_mode", 1),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )

    approx_split_weight_update = """
      // dummy computation to avoid unused variable warning
      weight_new.fma_(grad, -multiplier);
      assert(false); // approx rowwise AdaGrad is not supported on GPU
    """

    generate(
        optimizer="approx_rowwise_adagrad_with_weight_decay",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (FLOAT, "eps"),
                (FLOAT, "learning_rate"),
                (FLOAT, "weight_decay", 0.0),
                (INT, "weight_decay_mode", 1),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=approx_split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def rowwise_adagrad_with_counter() -> None:
    split_weight_update = """
        weight_new.acc.x = (exp_reg_correction * weight_new.acc.x - adjusted_multiplier * grad.acc.x);
        weight_new.acc.y = (exp_reg_correction * weight_new.acc.y - adjusted_multiplier * grad.acc.y);
        weight_new.acc.z = (exp_reg_correction * weight_new.acc.z - adjusted_multiplier * grad.acc.z);
        weight_new.acc.w = (exp_reg_correction * weight_new.acc.w - adjusted_multiplier * grad.acc.w);
    """
    split_precomputation = """
    at::acc_type<cache_t, true> freq = 1.0;
    at::acc_type<cache_t, true> l2_wd = 0.0;
    if (counter_halflife > 0 && threadIdx.x == 0) {
        // if id occurs multiple times in a batch, iter_delta=1
        const auto iter_delta = prev_iter[idx] == 0 ? 1.0 : iter * 1.0 - prev_iter[idx];
        prev_iter[idx] = iter * 1.0;
        const auto counter_log_rho = logf(2.0) / counter_halflife;
        row_counter[idx] = 1.0 + expf(-iter_delta * counter_log_rho) * row_counter[idx];
        freq = counter_halflife / row_counter[idx];
        if (weight_decay_mode == 1) {
            // L2 regularization
            l2_wd = 1.0;
        }
    }
    freq = shfl_sync(freq, 0);
    l2_wd = shfl_sync(l2_wd, 0);

    at::acc_type<cache_t, true> g_local_sum_square = 0.0;

    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
        Vec4T<at::acc_type<cache_t, true>> weight = weight_row_template.load(d, qparams_template);
        auto gx = grad_sum[i].acc.x + l2_wd * freq * weight_decay * weight.acc.x;
        auto gy = grad_sum[i].acc.y + l2_wd * freq * weight_decay * weight.acc.y;
        auto gz = grad_sum[i].acc.z + l2_wd * freq * weight_decay * weight.acc.z;
        auto gw = grad_sum[i].acc.w + l2_wd * freq * weight_decay * weight.acc.w;
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
    }

    const at::acc_type<cache_t, true> g_avg_square =
        warpReduceAllSum<at::acc_type<cache_t, true>>(g_local_sum_square) / D;

    at::acc_type<cache_t, true> multiplier;
    at::acc_type<cache_t, true> adjusted_multiplier;
    at::acc_type<cache_t, true> exp_reg_correction;
    at::acc_type<cache_t, true> tail_id_threshold_val = tail_id_threshold;
    if (is_tail_id_thresh_ratio == 1){
        tail_id_threshold_val = floorf(tail_id_threshold * max_counter);
    }
    if (threadIdx.x == 0) {
        at::acc_type<cache_t, true> new_sum_square_grads = momentum1[idx] + g_avg_square;
        momentum1[idx] = new_sum_square_grads;
        multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);

        adjusted_multiplier = multiplier;
        if ( learning_rate_mode >=0 ) {
            if (adjustment_iter <= 0 || (adjustment_iter > 0 && iter > adjustment_iter)) {

                if (row_counter[idx] > tail_id_threshold_val) {
                    if ( learning_rate_mode == 0 ) {
                        adjusted_multiplier = multiplier * max(min(powf(max_counter/(row_counter[idx] + 1.0), adjustment_ub), 10.0), 1.0);
                    } else if ( learning_rate_mode == 1 ) {
                        adjusted_multiplier = multiplier * min(max(powf((row_counter[idx] + 1.0)/max_counter, adjustment_ub), 0.1), 1.0);
                    } else if (learning_rate_mode == 2) {
                        adjusted_multiplier = learning_rate / (sqrtf(adjustment_ub*row_counter[idx]) + eps);
                    }
                }
            }
        }

        exp_reg_correction = 1.0;
        if (adjustment_iter <= 0 || (adjustment_iter > 0 && iter > adjustment_iter)) {
            if (weight_decay_mode == 2) {
                // Decoupled weight decay
                exp_reg_correction = 1.0 - freq * weight_decay * learning_rate;
            } else if (weight_decay_mode == 1) {
                // L2 regularization (coupled wd)
                exp_reg_correction = 1.0 - freq * weight_decay * multiplier;
            }
        }
    }
    multiplier = shfl_sync(multiplier, 0);
    adjusted_multiplier = shfl_sync(adjusted_multiplier, 0);
    exp_reg_correction = shfl_sync(exp_reg_correction, 0);
    """
    split_weight_update_cpu = """
        at::acc_type<grad_t, true> g_local_sum_square = 0.0;
        for (int64_t d = 0; d < D; ++d) {
            g_local_sum_square += grad_buffer[d] * grad_buffer[d];
        }
        auto g_avg_square = g_local_sum_square / D;
        auto offset_idx = momentum1_offsets_data[feature_begin] + idx;
        at::acc_type<grad_t, true> new_sum_square_grads = momentum1_host[offset_idx] + g_avg_square;
        momentum1_host[offset_idx] = new_sum_square_grads;
        at::acc_type<grad_t, true> multiplier;
        multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);
        const auto iter_delta = iter * 1.0 - prev_iter_host[offset_idx];
        prev_iter_host[offset_idx] = iter * 1.0;
        const auto exp_reg = 1.0 / (weight_decay * multiplier + 1.0);
        const auto exp_reg_correction = powf(exp_reg, iter_delta);
        for (int64_t d = 0; d < D; ++d) {
            const auto weight = host_weights_data[embedding_begin + d];
            host_weights_data[embedding_begin + d] = exp_reg_correction * weight - exp_reg * multiplier * grad_buffer[d];
        }
    """

    generate(
        optimizer="rowwise_adagrad_with_counter",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (TENSOR, "prev_iter"),
                (TENSOR, "row_counter"),
                (FLOAT, "eps"),
                (FLOAT, "learning_rate"),
                (FLOAT, "weight_decay", 0.0),
                (INT, "iter"),
                (INT, "counter_halflife", -1),
                (INT, "adjustment_iter", -1),
                (FLOAT, "adjustment_ub", 1.0),
                (INT, "learning_rate_mode", -1),
                (INT, "weight_decay_mode", 1),
                (INT, "grad_sum_decay", -1),
                (FLOAT, "max_counter"),
                (FLOAT, "tail_id_threshold", 0.0),
                (INT, "is_tail_id_thresh_ratio", 0),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )

    approx_split_weight_update = """
      // dummy computation to avoid unused variable warning
      weight_new.fma_(grad, -multiplier);
      assert(false); // approx rowwise AdaGrad is not supported on GPU
    """

    generate(
        optimizer="approx_rowwise_adagrad_with_counter",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (TENSOR, "prev_iter"),
                (TENSOR, "row_counter"),
                (FLOAT, "eps"),
                (FLOAT, "learning_rate"),
                (FLOAT, "weight_decay", 0.0),
                (INT, "iter"),
                (INT, "counter_halflife", -1),
                (INT, "adjustment_iter", -1),
                (FLOAT, "adjustment_ub", 1.0),
                (INT, "learning_rate_mode", -1),
                (INT, "weight_decay_mode", 1),
                (INT, "grad_sum_decay", -1),
                (FLOAT, "max_counter"),
                (FLOAT, "tail_id_threshold", 0.0),
                (INT, "is_tail_id_thresh_ratio", 0),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=approx_split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def rowwise_weighted_adagrad() -> None:
    split_weight_update = """
      weight_new.acc.x = correction * weight_new.acc.x - multiplier * grad.acc.x;
      weight_new.acc.y = correction * weight_new.acc.y - multiplier * grad.acc.y;
      weight_new.acc.z = correction * weight_new.acc.z - multiplier * grad.acc.z;
      weight_new.acc.w = correction * weight_new.acc.w - multiplier * grad.acc.w;
    """
    split_precomputation = """
    at::acc_type<cache_t, true> g_local_sum_square = 0.0;
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
        Vec4T<at::acc_type<cache_t, true>> weight = weight_row_template.load(d, qparams_template);
        auto gx = grad_sum[i].acc.x + weight_decay * weight.acc.x;
        auto gy = grad_sum[i].acc.y + weight_decay * weight.acc.y;
        auto gz = grad_sum[i].acc.z + weight_decay * weight.acc.z;
        auto gw = grad_sum[i].acc.w + weight_decay * weight.acc.w;
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
    }
    const at::acc_type<cache_t, true> g_avg_square =
        warpReduceAllSum<at::acc_type<cache_t, true>>(g_local_sum_square) / D;

    at::acc_type<cache_t, true> multiplier;
    at::acc_type<cache_t, true> correction;
    if (threadIdx.x == 0) {
        at::acc_type<cache_t, true> lambda = sqrtf(iter + 1);
        at::acc_type<cache_t, true> new_sum_square_grads = momentum1[idx] + lambda * g_avg_square;
        momentum1[idx] = new_sum_square_grads;
        multiplier = learning_rate * lambda / (cbrtf(new_sum_square_grads) + eps);
        correction = 1.0 - multiplier * weight_decay;
    }
    multiplier = shfl_sync(multiplier, 0);
    correction = shfl_sync(correction, 0);
    """
    split_weight_update_cpu = """
        // weight_decay not supported for cpu version
        at::acc_type<grad_t, true> g_local_sum_square = 0.0;
        for (int64_t d = 0; d < D; ++d) {
            g_local_sum_square += grad_buffer[d] * grad_buffer[d];
        }
        auto g_avg_square = g_local_sum_square / D;
        at::acc_type<grad_t, true> lambda = sqrtf(iter + 1);
        at::acc_type<grad_t, true> new_sum_square_grads = momentum1_host[momentum1_offsets_data[feature_begin] + idx] + lambda * g_avg_square;
        momentum1_host[momentum1_offsets_data[feature_begin] + idx] = new_sum_square_grads;
        at::acc_type<grad_t, true> multiplier;
        multiplier = learning_rate * lambda / (cbrtf(new_sum_square_grads) + eps);
        for (int64_t d = 0; d < D; ++d) {
            host_weights_data[embedding_begin + d] -= grad_buffer[d] * multiplier;
        }
    """

    generate(
        optimizer="rowwise_weighted_adagrad",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (FLOAT, "eps"),
                (FLOAT, "learning_rate"),
                (FLOAT, "weight_decay"),
                (INT, "iter"),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def sgd() -> None:
    split_weight_update = """
      weight_new.fma_(grad, -learning_rate);
    """
    split_weight_update_cpu = """
      for (int64_t d = 0; d < D; ++d) {
        host_weights_data[embedding_begin + d] -= learning_rate * grad_buffer[d];
      }
    """

    generate(
        optimizer="sgd",
        args=make_args([(FLOAT, "learning_rate")]),
        split_precomputation="",
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )

    approx_split_weight_update = """
      // approx_sgd not supported for GPU.
      // Just do the same thing as exact sgd to avoid unused variable warning.
      weight_new.fma_(grad, -learning_rate);
      assert(false); // approx SGD is not supported on GPU
    """

    generate(
        optimizer="approx_sgd",
        args=make_args([(FLOAT, "learning_rate")]),
        split_precomputation="",
        split_weight_update=approx_split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def lamb() -> None:
    split_precomputation = """
  at::acc_type<cache_t, true> weight_sum_sq = 0.0;
  at::acc_type<cache_t, true> rtw_sum_sq = 0.0;
  auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(weights, cache_weights, D, nullptr);
  float2 qparams;
  if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
    qparams = weight_row.load_qparams();
  }
#pragma unroll 1
  for (int32_t i = 0;
      i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
      ++i) {
    int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
    Vec4T<at::acc_type<cache_t, true>> weight = weight_row.load(d, qparams);
    Vec4T<at::acc_type<cache_t, true>> m1(&momentum1[idx * D + d]);

    m1.acc.x = beta1 * m1.acc.x + (1.0 - beta1) * grad_sum[i].acc.x;
    m1.acc.y = beta1 * m1.acc.y + (1.0 - beta1) * grad_sum[i].acc.y;
    m1.acc.z = beta1 * m1.acc.z + (1.0 - beta1) * grad_sum[i].acc.z;
    m1.acc.w = beta1 * m1.acc.w + (1.0 - beta1) * grad_sum[i].acc.w;
    m1.store(&momentum1[idx * D + d]);

    Vec4T<at::acc_type<cache_t, true>> m2(&momentum2[idx * D + d]);
    m2.acc.x = beta2 * m2.acc.x + (1.0 - beta2) * grad_sum[i].acc.x * grad_sum[i].acc.x;
    m2.acc.y = beta2 * m2.acc.y + (1.0 - beta2) * grad_sum[i].acc.y * grad_sum[i].acc.y;
    m2.acc.z = beta2 * m2.acc.z + (1.0 - beta2) * grad_sum[i].acc.z * grad_sum[i].acc.z;
    m2.acc.w = beta2 * m2.acc.w + (1.0 - beta2) * grad_sum[i].acc.w * grad_sum[i].acc.w;
    m2.store(&momentum2[idx * D + d]);

    // now, we are finished with grad_sum. We can *reuse* grad_sum to store r_t + weight_decay * weight;
    grad_sum[i].acc.x = (m1.acc.x / (1.0 - powf(beta1, iter))) / (sqrtf((m2.acc.x / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight.acc.x;
    grad_sum[i].acc.y = (m1.acc.y / (1.0 - powf(beta1, iter))) / (sqrtf((m2.acc.y / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight.acc.y;
    grad_sum[i].acc.z = (m1.acc.z / (1.0 - powf(beta1, iter))) / (sqrtf((m2.acc.z / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight.acc.z;
    grad_sum[i].acc.w = (m1.acc.w / (1.0 - powf(beta1, iter))) / (sqrtf((m2.acc.w / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight.acc.w;

    weight_sum_sq += weight.acc.x * weight.acc.x + weight.acc.y * weight.acc.y + weight.acc.z * weight.acc.z + weight.acc.w * weight.acc.w;
    rtw_sum_sq += grad_sum[i].acc.x * grad_sum[i].acc.x + grad_sum[i].acc.y * grad_sum[i].acc.y + grad_sum[i].acc.z * grad_sum[i].acc.z + grad_sum[i].acc.w * grad_sum[i].acc.w;
  }
  const auto weight_norm =
      sqrtf(warpReduceAllSum<at::acc_type<cache_t, true>>(weight_sum_sq));
  const auto rtw_norm =
      sqrtf(warpReduceAllSum<at::acc_type<cache_t, true>>(rtw_sum_sq));
   const auto true_ratio = weight_norm / rtw_norm;
"""
    split_weight_update = """
      weight_new.fma_(grad, -learning_rate * true_ratio);
    """
    split_weight_update_cpu = ""

    generate(
        optimizer="lamb",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (TENSOR, "momentum2"),
                (FLOAT, "learning_rate"),
                (FLOAT, "eps"),
                (FLOAT, "beta1"),
                (FLOAT, "beta2"),
                (FLOAT, "weight_decay"),
                (INT, "iter"),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def partial_rowwise_lamb() -> None:
    split_precomputation = """
    at::acc_type<cache_t, true> g_local_sum_square = 0.0;

    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
    g_local_sum_square += grad_sum[i].acc.x * grad_sum[i].acc.x +
        grad_sum[i].acc.y * grad_sum[i].acc.y +
        grad_sum[i].acc.z * grad_sum[i].acc.z +
        grad_sum[i].acc.w * grad_sum[i].acc.w;
    }
    const at::acc_type<cache_t, true> g_avg_square =
        warpReduceAllSum<at::acc_type<cache_t, true>>(g_local_sum_square) / D;

    at::acc_type<cache_t, true> m2;
    if (threadIdx.x == 0) {
        m2 = beta2 * momentum2[idx] + (1.0 - beta2) * g_avg_square;
        momentum2[idx] = m2;
    }
    m2 = shfl_sync(m2, 0);
    at::acc_type<cache_t, true> m2_hat = 1.0 / (sqrtf((m2 / (1.0 - powf(beta2, iter)))) + eps);

    at::acc_type<cache_t, true> weight_sum_sq = 0.0;
    at::acc_type<cache_t, true> rtw_sum_sq = 0.0;
    auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(weights, cache_weights, D, nullptr);
    float2 qparams;
    if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
        qparams = weight_row.load_qparams();
    }
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;

        Vec4T<at::acc_type<cache_t, true>> m1(&momentum1[idx * D + d]);
        m1.acc.x = beta1 * m1.acc.x + (1.0 - beta1) * grad_sum[i].acc.x;
        m1.acc.y = beta1 * m1.acc.y + (1.0 - beta1) * grad_sum[i].acc.y;
        m1.acc.z = beta1 * m1.acc.z + (1.0 - beta1) * grad_sum[i].acc.z;
        m1.acc.w = beta1 * m1.acc.w + (1.0 - beta1) * grad_sum[i].acc.w;
        m1.store(&momentum1[idx * D + d]);

        // now, we are finished with grad_sum. We can *reuse* grad_sum to store r_t + weight_decay * weight;
        Vec4T<at::acc_type<cache_t, true>> weight = weight_row.load(d, qparams);
        grad_sum[i].acc.x = (m1.acc.x / (1.0 - powf(beta1, iter))) * m2_hat + weight_decay * weight.acc.x;
        grad_sum[i].acc.y = (m1.acc.y / (1.0 - powf(beta1, iter))) * m2_hat + weight_decay * weight.acc.y;
        grad_sum[i].acc.z = (m1.acc.z / (1.0 - powf(beta1, iter))) * m2_hat + weight_decay * weight.acc.z;
        grad_sum[i].acc.w = (m1.acc.w / (1.0 - powf(beta1, iter))) * m2_hat + weight_decay * weight.acc.w;

        weight_sum_sq += weight.acc.x * weight.acc.x + weight.acc.y * weight.acc.y + weight.acc.z * weight.acc.z + weight.acc.w * weight.acc.w;
        rtw_sum_sq += grad_sum[i].acc.x * grad_sum[i].acc.x + grad_sum[i].acc.y * grad_sum[i].acc.y + grad_sum[i].acc.z * grad_sum[i].acc.z + grad_sum[i].acc.w * grad_sum[i].acc.w;
    }
    const auto weight_norm =
        sqrtf(warpReduceAllSum<at::acc_type<cache_t, true>>(weight_sum_sq));
    const auto rtw_norm =
        sqrtf(warpReduceAllSum<at::acc_type<cache_t, true>>(rtw_sum_sq));
    const auto true_ratio = weight_norm / rtw_norm;
    """

    split_weight_update = """
      weight_new.fma_(grad, -learning_rate * true_ratio);
    """
    split_weight_update_cpu = ""  # TODO

    generate(
        optimizer="partial_rowwise_lamb",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (TENSOR, "momentum2"),
                (FLOAT, "learning_rate"),
                (FLOAT, "eps"),
                (FLOAT, "beta1"),
                (FLOAT, "beta2"),
                (FLOAT, "weight_decay"),
                (INT, "iter"),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def adam() -> None:
    split_weight_update = """
      Vec4T<cache_t> m_t(&momentum1[idx * D + d]);
      m_t.acc.x *= beta1;
      m_t.acc.y *= beta1;
      m_t.acc.z *= beta1;
      m_t.acc.w *= beta1;
      m_t.fma_(grad, 1.0 - beta1);
      m_t.store(&momentum1[idx * D + d]);

      Vec4T<cache_t> v_t(&momentum2[idx * D + d]);
      v_t.acc.x *= beta2;
      v_t.acc.y *= beta2;
      v_t.acc.z *= beta2;
      v_t.acc.w *= beta2;

      grad.acc.x *= grad.acc.x;
      grad.acc.y *= grad.acc.y;
      grad.acc.z *= grad.acc.z;
      grad.acc.w *= grad.acc.w;
      v_t.fma_(grad, 1.0 - beta2);
      v_t.store(&momentum2[idx * D + d]);

      weight_new.acc.x -= learning_rate * (m_t.acc.x / (1.0 - powf(beta1, iter)) / (sqrtf((v_t.acc.x / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight_new.acc.x);
      weight_new.acc.y -= learning_rate * (m_t.acc.y / (1.0 - powf(beta1, iter)) / (sqrtf((v_t.acc.y / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight_new.acc.y);
      weight_new.acc.z -= learning_rate * (m_t.acc.z / (1.0 - powf(beta1, iter)) / (sqrtf((v_t.acc.z / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight_new.acc.z);
      weight_new.acc.w -= learning_rate * (m_t.acc.w / (1.0 - powf(beta1, iter)) / (sqrtf((v_t.acc.w / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight_new.acc.w);
    """
    split_weight_update_cpu = ""  # TODO

    generate(
        optimizer="adam",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (TENSOR, "momentum2"),
                (FLOAT, "learning_rate"),
                (FLOAT, "eps"),
                (FLOAT, "beta1"),
                (FLOAT, "beta2"),
                (FLOAT, "weight_decay"),
                (INT, "iter"),
            ]
        ),
        split_precomputation="",
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def partial_rowwise_adam() -> None:
    split_precomputation = """
    at::acc_type<cache_t, true> g_local_sum_square = 0.0;
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
    g_local_sum_square += grad_sum[i].acc.x * grad_sum[i].acc.x +
        grad_sum[i].acc.y * grad_sum[i].acc.y +
        grad_sum[i].acc.z * grad_sum[i].acc.z +
        grad_sum[i].acc.w * grad_sum[i].acc.w;
    }
    const at::acc_type<cache_t, true> g_avg_square =
        warpReduceAllSum<at::acc_type<cache_t, true>>(g_local_sum_square) / D;

    at::acc_type<cache_t, true> v_hat_t;
    if (threadIdx.x == 0) {
        at::acc_type<cache_t, true> v_t = momentum2[idx] * beta2 + g_avg_square * (1.0 - beta2);
        momentum2[idx] = v_t;
        v_hat_t = v_t / (1.0 - powf(beta2, iter));
    }
    v_hat_t = shfl_sync(v_hat_t, 0);
    """

    split_weight_update = """
      Vec4T<cache_t> m_t(&momentum1[idx * D + d]);
      m_t.acc.x *= beta1;
      m_t.acc.y *= beta1;
      m_t.acc.z *= beta1;
      m_t.acc.w *= beta1;
      m_t.fma_(grad, 1.0 - beta1);
      m_t.store(&momentum1[idx * D + d]);

      weight_new.acc.x -= learning_rate * (m_t.acc.x / (1.0 - powf(beta1, iter)) / (sqrtf(v_hat_t) + eps) + weight_decay * weight_new.acc.x);
      weight_new.acc.y -= learning_rate * (m_t.acc.y / (1.0 - powf(beta1, iter)) / (sqrtf(v_hat_t) + eps) + weight_decay * weight_new.acc.y);
      weight_new.acc.z -= learning_rate * (m_t.acc.z / (1.0 - powf(beta1, iter)) / (sqrtf(v_hat_t) + eps) + weight_decay * weight_new.acc.z);
      weight_new.acc.w -= learning_rate * (m_t.acc.w / (1.0 - powf(beta1, iter)) / (sqrtf(v_hat_t) + eps) + weight_decay * weight_new.acc.w);
    """
    split_weight_update_cpu = ""  # TODO

    generate(
        optimizer="partial_rowwise_adam",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (TENSOR, "momentum2"),
                (FLOAT, "learning_rate"),
                (FLOAT, "eps"),
                (FLOAT, "beta1"),
                (FLOAT, "beta2"),
                (FLOAT, "weight_decay"),
                (INT, "iter"),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def lars_sgd() -> None:
    split_precomputation = """
  at::acc_type<cache_t, true> weight_sum_sq = 0.0;
  at::acc_type<cache_t, true> grad_sum_sq = 0.0;

  auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(weights, cache_weights, D, nullptr);
  float2 qparams;
  if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
      qparams = weight_row.load_qparams();
  }
#pragma unroll kMaxVecsPerThread
  for (int32_t i = 0;
      i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
      ++i) {
    int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
    Vec4T<at::acc_type<cache_t,true>> weight = weight_row.load(d, qparams);
    weight_sum_sq += weight.acc.x * weight.acc.x + weight.acc.y * weight.acc.y + weight.acc.z * weight.acc.z + weight.acc.w * weight.acc.w;
    grad_sum_sq += grad_sum[i].acc.x * grad_sum[i].acc.x + grad_sum[i].acc.y * grad_sum[i].acc.y + grad_sum[i].acc.z * grad_sum[i].acc.z + grad_sum[i].acc.w * grad_sum[i].acc.w;
  }
  const auto weight_norm =
      sqrtf(warpReduceAllSum<at::acc_type<cache_t, true>>(weight_sum_sq));
  const auto grad_norm =
      sqrtf(warpReduceAllSum<at::acc_type<cache_t, true>>(grad_sum_sq));
   const at::acc_type<cache_t, true> adjusted_lr = learning_rate * eta * weight_norm / (grad_norm + weight_decay * weight_norm);
"""

    split_weight_update = """
      Vec4T<cache_t> m1(&momentum1[idx * D + d]);
      m1.acc.x = momentum * m1.acc.x + adjusted_lr * (grad.acc.x + weight_decay * weight_new.acc.x);
      m1.acc.y = momentum * m1.acc.y + adjusted_lr * (grad.acc.y + weight_decay * weight_new.acc.y);
      m1.acc.z = momentum * m1.acc.z + adjusted_lr * (grad.acc.z + weight_decay * weight_new.acc.z);
      m1.acc.w = momentum * m1.acc.w + adjusted_lr * (grad.acc.w + weight_decay * weight_new.acc.w);
      m1.store(&momentum1[idx * D + d]);

      weight_new.acc.x -= m1.acc.x;
      weight_new.acc.y -= m1.acc.y;
      weight_new.acc.z -= m1.acc.z;
      weight_new.acc.w -= m1.acc.w;
    """
    split_weight_update_cpu = ""  # TODO

    generate(
        optimizer="lars_sgd",
        args=make_args(
            [
                (TENSOR, "momentum1"),
                (FLOAT, "learning_rate"),
                (FLOAT, "eta"),
                (FLOAT, "momentum"),
                (FLOAT, "weight_decay"),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=split_weight_update,
        split_weight_update_cpu=split_weight_update_cpu,
    )


def forward_split() -> None:
    template = env.get_template("embedding_forward_split_template.cu")

    src_cu = template.render(weighted=False)
    write("gen_embedding_forward_split_unweighted_codegen_cuda.cu", src_cu)
    src_cu = template.render(weighted=True)
    write("gen_embedding_forward_split_weighted_codegen_cuda.cu", src_cu)

    src_cu = template.render(weighted=False, dense=True)
    write("gen_embedding_forward_dense_unweighted_codegen_cuda.cu", src_cu)
    src_cu = template.render(weighted=True, dense=True)
    write("gen_embedding_forward_dense_weighted_codegen_cuda.cu", src_cu)


def forward_quantized() -> None:
    @dataclass
    class elem_type:
        enum_name: str
        cpp_type_name: str
        primitive_type: str
        bit_width: int

    type_map = {
        "FP32": elem_type("FP32", "float", "FP", 32),
        "FP16": elem_type("FP16", "__half2", "FP", 16),
        "FP8": elem_type("FP8", "uint32_t", "FP", 8),
        "INT8": elem_type("INT8", "uint32_t", "INT", 8),
        "INT4": elem_type("INT4", "uint32_t", "INT", 4),
        "INT2": elem_type("INT2", "uint32_t", "INT", 2),
    }

    template = env.get_template("embedding_forward_quantized_split_template.cu")
    src_cu = template.render(weighted=False, type_map=type_map)
    write("gen_embedding_forward_quantized_split_unweighted_codegen_cuda.cu", src_cu)
    src_cu = template.render(weighted=True, type_map=type_map)
    write("gen_embedding_forward_quantized_split_weighted_codegen_cuda.cu", src_cu)

    template = env.get_template("embedding_forward_quantized_cpu_template.cpp")
    src_cu = template.render(weighted=False, type_map=type_map)
    write("gen_embedding_forward_quantized_unweighted_codegen_cpu.cpp", src_cu)
    src_cu = template.render(weighted=True, type_map=type_map)
    write("gen_embedding_forward_quantized_weighted_codegen_cpu.cpp", src_cu)


def backward_indices() -> None:
    template = env.get_template("embedding_backward_split_indice_weights_template.cu")
    src_cu = template.render()
    write("gen_embedding_backward_split_indice_weights_codegen_cuda.cu", src_cu)
    src_cu = template.render(dense=True)
    write("gen_embedding_backward_dense_indice_weights_codegen_cuda.cu", src_cu)


def backward_dense() -> None:
    generate(
        optimizer="dense",
        dense=True,
        args=make_args(
            [
                (FLOAT, "unused"),
            ]
        ),
    )


def gen__init__py() -> None:
    template = env.get_template("__init__.template")
    src_py = template.render()
    write("__init__.py", src_py)


def emb_codegen(
    install_dir: Optional[str] = None, is_fbcode: Optional[bool] = None
) -> None:
    if install_dir is not None and len(install_dir) != 0:
        args.install_dir = install_dir
    if is_fbcode is not None:
        args.is_fbcode = is_fbcode
    adagrad()
    adam()
    backward_indices()
    backward_dense()
    forward_quantized()
    forward_split()
    lamb()
    lars_sgd()
    partial_rowwise_adam()
    partial_rowwise_lamb()
    rowwise_adagrad()
    rowwise_adagrad_with_weight_decay()
    rowwise_adagrad_with_counter()
    rowwise_weighted_adagrad()
    sgd()

    gen__init__py()


def main() -> None:
    emb_codegen()


if __name__ == "__main__":
    main()
