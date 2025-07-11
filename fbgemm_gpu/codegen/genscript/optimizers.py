#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

from typing import Any, Dict

try:
    from .jinja_environment import generate_optimized_grad_sum_loop_access
    from .optimizer_args import OptimizerArgsSet, OptimizerArgsSetItem as OptimItem
    from .torch_type_utils import ArgType
except:
    # pyre-ignore[21]
    from jinja_environment import generate_optimized_grad_sum_loop_access

    # pyre-ignore[21]
    from optimizer_args import OptimizerArgsSet, OptimizerArgsSetItem as OptimItem

    # pyre-ignore[21]
    from torch_type_utils import ArgType

######################################################################
## Optimizer templates                                              ##
######################################################################


def dense() -> Dict[str, Any]:
    return {
        "optimizer": "dense",
        "dense": True,
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.FLOAT, "unused"),
            ],
            {
                "v1": "float unused = 0",
            },
        ),
        "has_cpu_support": True,
        "has_gpu_support": True,
        "has_vbe_support": True,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


def adagrad() -> Dict[str, Any]:
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

    return {
        "optimizer": "adagrad",
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
            ],
            {"v1": "Tensor momentum1, float eps = 0, float learning_rate = 0"},
        ),
        "split_precomputation": "",
        "split_weight_update": split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": True,
        "has_gpu_support": True,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


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


def rowwise_adagrad() -> Dict[str, Any]:
    split_weight_update = """
        weight_new.acc.x = correction * weight_new.acc.x - multiplier * grad.acc.x;
        weight_new.acc.y = correction * weight_new.acc.y - multiplier * grad.acc.y;
        weight_new.acc.z = correction * weight_new.acc.z - multiplier * grad.acc.z;
        weight_new.acc.w = correction * weight_new.acc.w - multiplier * grad.acc.w;
    """
    split_post_update = """
    if (max_norm > 0.0) {
        CUDA_KERNEL_ASSERT(!(std::is_same<emb_t, uint8_t>::value && !cache_weights)); // not supported for uint8 yet

        // compute weight norm
        at::acc_type<cache_t, true> weight_sum_square = 0.0;
        for (int32_t vec = 0;
             vec < max_vecs && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
             ++vec) {
            const int32_t d = (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH;
            Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);
            weight_sum_square
                += weight_new.acc.x * weight_new.acc.x
                + weight_new.acc.y * weight_new.acc.y
                + weight_new.acc.z * weight_new.acc.z
                + weight_new.acc.w * weight_new.acc.w;
        }
        const at::acc_type<cache_t, true> weight_norm =
            sqrtf(GROUP_REDUCE_ALL_SUM(weight_sum_square, at::acc_type<cache_t, true>));

        // scale by max_norm if weight_norm exceeds max_norm
        if (threadIdx.x == 0) {
            multiplier = weight_norm > max_norm ? max_norm / weight_norm : 1.0f;
        }
        multiplier = SHFL_SYNC(multiplier, 0);
        if (weight_norm > max_norm) {
            for (int32_t vec = 0;
                 vec < max_vecs && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
                 ++vec) {
                const int32_t d = (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH;
                Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);

                weight_new.acc.x *= multiplier;
                weight_new.acc.y *= multiplier;
                weight_new.acc.z *= multiplier;
                weight_new.acc.w *= multiplier;
                weight_row_template.store(weight_new, d, qparams_new); // qparams_new not used if embedding is not int8
            }
        }
    }
    """
    split_precomputation = """
    at::acc_type<cache_t, true> g_local_sum_square = 0.0;
    """
    split_precomputation += generate_optimized_grad_sum_loop_access(
        """
        const float4* grad = &{grad_vec}.acc;
        auto gx = grad->x;
        auto gy = grad->y;
        auto gz = grad->z;
        auto gw = grad->w;
        if (weight_decay_mode == 1) {
            // L2 regularization
            Vec4TAcc<cache_t> weight = weight_row_template.load(d, qparams_template);
            gx += weight_decay * weight.acc.x;
            gy += weight_decay * weight.acc.y;
            gz += weight_decay * weight.acc.z;
            gw += weight_decay * weight.acc.w;
        }
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
    """
    )
    split_precomputation += """	
	// Define the rowwise adagrad optimizer state struct view
    struct [[maybe_unused]] OptimizerState {
        at::acc_type<cache_t, true> momentum;
    };

    const at::acc_type<cache_t, true> g_avg_square =
        GROUP_REDUCE_ALL_SUM(g_local_sum_square, at::acc_type<cache_t, true>) / D;

    at::acc_type<cache_t, true> multiplier = 0.0;
    at::acc_type<cache_t, true> correction = 0.0;
    if (threadIdx.x == 0) {	
        auto new_sum_square_grads = g_avg_square;
	
        // Update the optimizer state.  Use optimizer state offloading only if 
        // SSD and if enabled by the user
        if (enable_optimizer_offloading) {
            // Fetch the pointer to the optimizer state along the cache row
            auto* optimizer = weight_row_template.template optimizer_state_ptr<OptimizerState>();
            new_sum_square_grads += optimizer->momentum;
            optimizer->momentum = new_sum_square_grads;
        
        } else {
            new_sum_square_grads += momentum1[idx];
            momentum1[idx] = new_sum_square_grads;
        }

        multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);
        if (weight_decay_mode == 1) {
            // L2 regularization
            correction = 1.0 - multiplier * weight_decay;
        } else if (weight_decay_mode == 2 || weight_decay_mode == 5) {
            // Decoupled weight decay
            correction = 1.0 - learning_rate * weight_decay;
        } else {
            // default value
            correction = 1.0;
        }
    }
    multiplier = SHFL_SYNC(multiplier, 0);
    correction = SHFL_SYNC(correction, 0);
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
        } else if (weight_decay_mode == 2 || weight_decay_mode == 5) {
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

    return {
        "optimizer": "rowwise_adagrad",
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "weight_decay", 0.0),
                OptimItem(ArgType.INT, "weight_decay_mode", 0),
                OptimItem(ArgType.FLOAT, "max_norm", 0.0),
            ],
            {
                "v1": "Tensor momentum1, float eps = 0, float learning_rate = 0, float weight_decay = 0.0, int weight_decay_mode = 0.0, float max_norm = 0.0"
            },
        ),
        "split_precomputation": split_precomputation,
        "split_weight_update": split_weight_update,
        "split_post_update": split_post_update,
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": True,
        "has_gpu_support": True,
        "has_vbe_support": True,
        "has_global_weight_decay_support": True,
        "has_ssd_support": True,
    }


def approx_rowwise_adagrad() -> Dict[str, Any]:
    rowwise_adagrad_args = rowwise_adagrad()

    approx_split_weight_update = """
      // dummy computation to avoid unused variable warning
      weight_new.fma_(grad, -multiplier);
      assert(false); // approx rowwise AdaGrad is not supported on GPU
    """

    return {
        "optimizer": "approx_rowwise_adagrad",
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "weight_decay", 0.0),
                OptimItem(ArgType.INT, "weight_decay_mode", 0),
            ],
            {
                "v1": "Tensor momentum1, float eps = 0, float learning_rate = 0, float weight_decay = 0.0, int weight_decay_mode = 0.0",
            },
        ),
        "split_precomputation": rowwise_adagrad_args["split_precomputation"],
        "split_weight_update": approx_split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": rowwise_adagrad_args["split_weight_update_cpu"],
        "has_cpu_support": False,
        "has_gpu_support": False,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


# Deprecated, to be cleaned up
def rowwise_adagrad_with_weight_decay() -> Dict[str, Any]:
    split_weight_update = """
        weight_new.acc.x = correction * weight_new.acc.x - multiplier * grad.acc.x;
        weight_new.acc.y = correction * weight_new.acc.y - multiplier * grad.acc.y;
        weight_new.acc.z = correction * weight_new.acc.z - multiplier * grad.acc.z;
        weight_new.acc.w = correction * weight_new.acc.w - multiplier * grad.acc.w;
    """
    split_precomputation = """
    at::acc_type<cache_t, true> g_local_sum_square = 0.0;
    """
    split_precomputation += generate_optimized_grad_sum_loop_access(
        """
        const float4* grad = &{grad_vec}.acc;
        auto gx = grad->x;
        auto gy = grad->y;
        auto gz = grad->z;
        auto gw = grad->w;
        if (weight_decay_mode == 1) {
            // L2 regularization
            Vec4TAcc<cache_t> weight = weight_row_template.load(d, qparams_template);
            gx += weight_decay * weight.acc.x;
            gy += weight_decay * weight.acc.y;
            gz += weight_decay * weight.acc.z;
            gw += weight_decay * weight.acc.w;
        }
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
    """
    )
    split_precomputation += """
    const at::acc_type<cache_t, true> g_avg_square =
        GROUP_REDUCE_ALL_SUM(g_local_sum_square, at::acc_type<cache_t, true>) / D;

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
    multiplier = SHFL_SYNC(multiplier, 0);
    correction = SHFL_SYNC(correction, 0);
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

    return {
        "optimizer": "rowwise_adagrad_with_weight_decay",
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "weight_decay", 0.0),
                OptimItem(ArgType.INT, "weight_decay_mode", 0),
            ],
            {
                "v1": "Tensor momentum1, float eps = 0, float learning_rate = 0, float weight_decay = 0.0, int weight_decay_mode = 0.0"
            },
        ),
        "split_precomputation": split_precomputation,
        "split_weight_update": split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": False,
        "has_gpu_support": False,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


# Deprecated, to be cleaned up
def approx_rowwise_adagrad_with_weight_decay() -> Dict[str, Any]:
    rowwise_adagrad_with_weight_decay_args = rowwise_adagrad_with_weight_decay()

    approx_split_weight_update = """
      // dummy computation to avoid unused variable warning
      weight_new.fma_(grad, -multiplier);
      assert(false); // approx rowwise AdaGrad is not supported on GPU
    """

    return {
        "optimizer": "approx_rowwise_adagrad_with_weight_decay",
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "weight_decay", 0.0),
                OptimItem(ArgType.INT, "weight_decay_mode", 0),
            ],
            {
                "v1": "Tensor momentum1, float eps = 0, float learning_rate = 0, float weight_decay = 0.0, int weight_decay_mode = 0.0"
            },
        ),
        "split_precomputation": rowwise_adagrad_with_weight_decay_args[
            "split_precomputation"
        ],
        "split_weight_update": approx_split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": rowwise_adagrad_with_weight_decay_args[
            "split_weight_update_cpu"
        ],
        "has_cpu_support": False,
        "has_gpu_support": False,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


def rowwise_adagrad_with_counter() -> Dict[str, Any]:
    split_weight_update = """
        weight_new.acc.x = (exp_reg_correction * weight_new.acc.x - adjusted_multiplier * grad.acc.x);
        weight_new.acc.y = (exp_reg_correction * weight_new.acc.y - adjusted_multiplier * grad.acc.y);
        weight_new.acc.z = (exp_reg_correction * weight_new.acc.z - adjusted_multiplier * grad.acc.z);
        weight_new.acc.w = (exp_reg_correction * weight_new.acc.w - adjusted_multiplier * grad.acc.w);
    """
    split_precomputation = """
    at::acc_type<cache_t, true> freq = 1.0;
    at::acc_type<cache_t, true> tail_id_threshold_val = tail_id_threshold;
    CUDA_KERNEL_ASSERT(max_counter != 0.0); // avoid divide by zero error
    if (is_tail_id_thresh_ratio == 1){
        tail_id_threshold_val = floorf(tail_id_threshold * max_counter);
    }
    if (threadIdx.x == 0) {
        if (counter_halflife > 0) { // decay based on counter_halflife
            // if id occurs multiple times in a batch, iter_delta=1
            const auto iter_delta = prev_iter[idx] == 0 ? 1.0 : iter * 1.0 - prev_iter[idx];
            const auto counter_log_rho = logf(2.0) / counter_halflife;
            row_counter[idx] = 1.0 + expf(-iter_delta * counter_log_rho) * row_counter[idx];
        } else if (counter_halflife == 0) { // count only 1 (appear or not)
            row_counter[idx] = 1.0;
        } else { // count raw appearance without decaying
            row_counter[idx] += 1.0;
        }
        freq = counter_halflife / row_counter[idx];
    }
    freq = SHFL_SYNC(freq, 0);
    tail_id_threshold_val = SHFL_SYNC(tail_id_threshold_val, 0);

    at::acc_type<cache_t, true> g_local_sum_square = 0.0;
    at::acc_type<cache_t, true> w_local_sum_square = 0.0;
    """
    split_precomputation += generate_optimized_grad_sum_loop_access(
        """
        const float4* grad = &{grad_vec}.acc;
        auto gx = grad->x;
        auto gy = grad->y;
        auto gz = grad->z;
        auto gw = grad->w;

        Vec4TAcc<cache_t> weight = weight_row_template.load(d, qparams_template);

        // for L2 regularization (weight_decay_mode=1)
        // add weight_decay to gradient before other computation
        if (weight_decay_mode == 1) {
            gx += weight_decay * weight.acc.x;
            gy += weight_decay * weight.acc.y;
            gz += weight_decay * weight.acc.z;
            gw += weight_decay * weight.acc.w;
        }
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;

        // cow_clip (regularization_mode=4) requires weight norm
        if (regularization_mode == 4) {
            w_local_sum_square += weight.acc.x * weight.acc.x + weight.acc.y * weight.acc.y + weight.acc.z * weight.acc.z + weight.acc.w * weight.acc.w;
        }
    """
    )
    split_precomputation += """
    const at::acc_type<cache_t, true> g_sum_square =
        GROUP_REDUCE_ALL_SUM(g_local_sum_square, at::acc_type<cache_t, true>);
    const at::acc_type<cache_t, true> g_avg_square = g_sum_square / D;
    const at::acc_type<cache_t, true> w_sum_square =
        GROUP_REDUCE_ALL_SUM(w_local_sum_square, at::acc_type<cache_t, true>);

    at::acc_type<cache_t, true> adjusted_multiplier;
    at::acc_type<cache_t, true> exp_reg_correction;
    adjusted_multiplier = 0.0;
    exp_reg_correction = 0.0;

    if (threadIdx.x == 0) {
        at::acc_type<cache_t, true> new_sum_square_grads = momentum1[idx] + g_avg_square;
        momentum1[idx] = new_sum_square_grads;
        const auto multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);
        const auto adjustment_enabled = adjustment_iter <= 0 || (adjustment_iter > 0 && iter > adjustment_iter);

        if (regularization_mode == 3) { // counter-based regularization (regularization_mode=3)
            adjusted_multiplier = multiplier;
            if (learning_rate_mode >=0 && adjustment_enabled) {
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
        } else if (regularization_mode == 4) { // cow-clip (regularization_mode=4)
            const auto clip_thresh = row_counter[idx] * max(weight_norm_coefficient * sqrtf(w_sum_square), lower_bound);
            adjusted_multiplier = min(1.0f, clip_thresh / sqrtf(g_sum_square)) * multiplier;
        }

        exp_reg_correction = 1.0;
        if (regularization_mode == 3) { // counter-based regularization (regularization_mode=3)
            if (adjustment_enabled) {
               if (weight_decay_mode == 3) { // AdagradW (weight_decay_mode=3)
                    if (counter_halflife < 0) {
                        adjusted_multiplier = multiplier * sqrtf(row_counter[idx] * 1.0);
                        exp_reg_correction = 1.0 - weight_decay * learning_rate;
                        const auto lazy_delta = prev_iter[idx] == 0 ? 1.0 : iter * 1.0 - prev_iter[idx];
                        const auto lazy_multiplier = powf(exp_reg_correction, min(lazy_delta, iter * 1.0 - adjustment_iter) - 1.0);
                        adjusted_multiplier *= lazy_multiplier;
                        exp_reg_correction *= lazy_multiplier;
                    }
                } else if (weight_decay_mode == 2) { // Decoupled weight decay (weight_decay_mode=2)
                    exp_reg_correction = 1.0 - freq * weight_decay * learning_rate;
                } else if (weight_decay_mode == 1) { // L2 regularization (coupled wd)
                    exp_reg_correction = 1.0 - freq * weight_decay * multiplier;
                }
            }
        } else if (regularization_mode == 4) { // cow-clip (regularization_mode=4)
            if (weight_decay_mode == 2) { // Decoupled weight decay (weight_decay_mode=2)
                exp_reg_correction = 1.0 -  weight_decay * learning_rate;
            } else if (weight_decay_mode == 1) { // L2 regularization (coupled wd)
                exp_reg_correction = 1.0 - weight_decay * adjusted_multiplier;
            }
        }
        prev_iter[idx] = iter * 1.0;
    }
    adjusted_multiplier = SHFL_SYNC(adjusted_multiplier, 0);
    exp_reg_correction = SHFL_SYNC(exp_reg_correction, 0);
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

    return {
        "optimizer": "rowwise_adagrad_with_counter",
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "prev_iter"),
                OptimItem(ArgType.TENSOR, "row_counter"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "weight_decay", 0.0),
                OptimItem(ArgType.INT, "iter"),
                OptimItem(ArgType.INT, "counter_halflife", -1),
                OptimItem(ArgType.INT, "adjustment_iter", -1),
                OptimItem(ArgType.FLOAT, "adjustment_ub", 1.0),
                OptimItem(ArgType.INT, "learning_rate_mode", -1),
                OptimItem(ArgType.INT, "weight_decay_mode", 1),
                OptimItem(ArgType.INT, "grad_sum_decay", -1),
                OptimItem(ArgType.FLOAT, "max_counter"),
                OptimItem(ArgType.FLOAT, "tail_id_threshold", 0.0),
                OptimItem(ArgType.INT, "is_tail_id_thresh_ratio", 0),
                OptimItem(ArgType.INT, "regularization_mode", 0),
                OptimItem(ArgType.FLOAT, "weight_norm_coefficient", 0.0),
                OptimItem(ArgType.FLOAT, "lower_bound", 0.0),
            ],
            {
                "v1": "Tensor momentum1, Tensor prev_iter, Tensor row_counter, float eps = 0, float learning_rate = 0, float weight_decay = 0.0, int iter = 0, int counter_halflife = -1, int adjustment_iter = -1, float adjustment_ub = 1.0, int learning_rate_mode = -1, int weight_decay_mode = 1, int grad_sum_decay = -1, float max_counter = 0, float tail_id_threshold = 0.0, int is_tail_id_thresh_ratio = 0, int regularization_mode = 0, float weight_norm_coefficient = 0.0, float lower_bound = 0.0"
            },
        ),
        "split_precomputation": split_precomputation,
        "split_weight_update": split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": False,
        "has_gpu_support": True,
        "has_vbe_support": True,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


def approx_rowwise_adagrad_with_counter() -> Dict[str, Any]:
    rowwise_adagrad_with_counter_args = rowwise_adagrad_with_counter()

    approx_split_weight_update = """
      // dummy computation to avoid unused variable warning
      weight_new.fma_(grad, -learning_rate);
      assert(false); // approx rowwise AdaGrad is not supported on GPU
    """

    return {
        "optimizer": "approx_rowwise_adagrad_with_counter",
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "prev_iter"),
                OptimItem(ArgType.TENSOR, "row_counter"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "weight_decay", 0.0),
                OptimItem(ArgType.INT, "iter"),
                OptimItem(ArgType.INT, "counter_halflife", -1),
                OptimItem(ArgType.INT, "adjustment_iter", -1),
                OptimItem(ArgType.FLOAT, "adjustment_ub", 1.0),
                OptimItem(ArgType.INT, "learning_rate_mode", -1),
                OptimItem(ArgType.INT, "weight_decay_mode", 1),
                OptimItem(ArgType.INT, "grad_sum_decay", -1),
                OptimItem(ArgType.FLOAT, "max_counter"),
                OptimItem(ArgType.FLOAT, "tail_id_threshold", 0.0),
                OptimItem(ArgType.INT, "is_tail_id_thresh_ratio", 0),
                OptimItem(ArgType.INT, "regularization_mode", 0),
                OptimItem(ArgType.FLOAT, "weight_norm_coefficient", 0.0),
                OptimItem(ArgType.FLOAT, "lower_bound", 0.0),
            ],
            {
                "v1": "Tensor momentum1, Tensor prev_iter, Tensor row_counter, float eps = 0, float learning_rate = 0, float weight_decay = 0.0, int iter = 0, int counter_halflife = -1, int adjustment_iter = -1, float adjustment_ub = 1.0, int learning_rate_mode = -1, int weight_decay_mode = 1, int grad_sum_decay = -1, float max_counter = 0, float tail_id_threshold = 0.0, int is_tail_id_thresh_ratio = 0, int regularization_mode = 0, float weight_norm_coefficient = 0.0, float lower_bound = 0.0"
            },
        ),
        "split_precomputation": rowwise_adagrad_with_counter_args[
            "split_precomputation"
        ],
        "split_weight_update": approx_split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": rowwise_adagrad_with_counter_args[
            "split_weight_update_cpu"
        ],
        "has_cpu_support": False,
        "has_gpu_support": False,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


# Deprecated, to be cleaned up
def rowwise_weighted_adagrad() -> Dict[str, Any]:
    split_weight_update = """
      weight_new.acc.x = correction * weight_new.acc.x - multiplier * grad.acc.x;
      weight_new.acc.y = correction * weight_new.acc.y - multiplier * grad.acc.y;
      weight_new.acc.z = correction * weight_new.acc.z - multiplier * grad.acc.z;
      weight_new.acc.w = correction * weight_new.acc.w - multiplier * grad.acc.w;
    """
    split_precomputation = """
    at::acc_type<cache_t, true> g_local_sum_square = 0.0;
    """
    split_precomputation += generate_optimized_grad_sum_loop_access(
        """
        const float4* grad = &{grad_vec}.acc;
        Vec4TAcc<cache_t> weight = weight_row_template.load(d, qparams_template);
        auto gx = grad->x + weight_decay * weight.acc.x;
        auto gy = grad->y + weight_decay * weight.acc.y;
        auto gz = grad->z + weight_decay * weight.acc.z;
        auto gw = grad->w + weight_decay * weight.acc.w;
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
    """
    )
    split_precomputation += """
    const at::acc_type<cache_t, true> g_avg_square =
        GROUP_REDUCE_ALL_SUM(g_local_sum_square, at::acc_type<cache_t, true>) / D;

    at::acc_type<cache_t, true> multiplier;
    at::acc_type<cache_t, true> correction;
    if (threadIdx.x == 0) {
        at::acc_type<cache_t, true> lambda = sqrtf(iter + 1);
        at::acc_type<cache_t, true> new_sum_square_grads = momentum1[idx] + lambda * g_avg_square;
        momentum1[idx] = new_sum_square_grads;
        multiplier = learning_rate * lambda / (cbrtf(new_sum_square_grads) + eps);
        correction = 1.0 - multiplier * weight_decay;
    }
    multiplier = SHFL_SYNC(multiplier, 0);
    correction = SHFL_SYNC(correction, 0);
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

    return {
        "optimizer": "rowwise_weighted_adagrad",
        "is_experimental_optimizer": True,
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "weight_decay"),
                OptimItem(ArgType.INT, "iter"),
            ],
            {
                "v1": "Tensor momentum1, float eps = 0, float learning_rate = 0, float weight_decay = 0, int iter = 0"
            },
        ),
        "split_precomputation": split_precomputation,
        "split_weight_update": split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": False,
        "has_gpu_support": False,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


def sgd() -> Dict[str, Any]:
    split_weight_update = """
      weight_new.fma_(grad, -learning_rate);
    """
    split_weight_update_cpu = """
      for (int64_t d = 0; d < D; ++d) {
        host_weights_data[embedding_begin + d] -= learning_rate * grad_buffer[d];
      }
    """

    return {
        "optimizer": "sgd",
        "args": OptimizerArgsSet.create(
            [OptimItem(ArgType.TENSOR, "learning_rate_tensor")],
            {"v1": "float learning_rate = 0"},
        ),
        "split_precomputation": "",
        "split_weight_update": split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": True,
        "has_gpu_support": True,
        "has_vbe_support": True,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


def approx_sgd() -> Dict[str, Any]:
    sgd_args = sgd()

    approx_split_weight_update = """
      // approx_sgd not supported for GPU.
      // Just do the same thing as exact sgd to avoid unused variable warning.
      weight_new.fma_(grad, -learning_rate);
      assert(false); // approx SGD is not supported on GPU
    """

    return {
        "optimizer": "approx_sgd",
        "args": OptimizerArgsSet.create(
            [OptimItem(ArgType.TENSOR, "learning_rate_tensor")],
            {"v1": "float learning_rate = 0"},
        ),
        "split_precomputation": "",
        "split_weight_update": approx_split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": sgd_args["split_weight_update_cpu"],
        "has_cpu_support": False,
        "has_gpu_support": False,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


def lamb() -> Dict[str, Any]:
    split_precomputation = """
    at::acc_type<cache_t, true> weight_sum_sq = 0.0;
    at::acc_type<cache_t, true> rtw_sum_sq = 0.0;
    auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(weights, cache_weights, D);
    float2 qparams;
    if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
      qparams = weight_row.load_qparams();
    }
    """
    split_precomputation += generate_optimized_grad_sum_loop_access(
        """
      float4* grad = &{grad_vec}.acc;

      Vec4TAcc<cache_t> weight = weight_row.load(d, qparams);
      Vec4TAcc<cache_t> m1(&momentum1[idx * D + d]);

      m1.acc.x = beta1 * m1.acc.x + (1.0 - beta1) * grad->x;
      m1.acc.y = beta1 * m1.acc.y + (1.0 - beta1) * grad->y;
      m1.acc.z = beta1 * m1.acc.z + (1.0 - beta1) * grad->z;
      m1.acc.w = beta1 * m1.acc.w + (1.0 - beta1) * grad->w;
      m1.store(&momentum1[idx * D + d]);

      Vec4TAcc<cache_t> m2(&momentum2[idx * D + d]);
      m2.acc.x = beta2 * m2.acc.x + (1.0 - beta2) * grad->x * grad->x;
      m2.acc.y = beta2 * m2.acc.y + (1.0 - beta2) * grad->y * grad->y;
      m2.acc.z = beta2 * m2.acc.z + (1.0 - beta2) * grad->z * grad->z;
      m2.acc.w = beta2 * m2.acc.w + (1.0 - beta2) * grad->w * grad->w;
      m2.store(&momentum2[idx * D + d]);

      // now, we are finished with grad_sum. We can *reuse* grad_sum to store r_t + weight_decay * weight;
      grad->x = (m1.acc.x / (1.0 - powf(beta1, iter))) / (sqrtf((m2.acc.x / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight.acc.x;
      grad->y = (m1.acc.y / (1.0 - powf(beta1, iter))) / (sqrtf((m2.acc.y / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight.acc.y;
      grad->z = (m1.acc.z / (1.0 - powf(beta1, iter))) / (sqrtf((m2.acc.z / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight.acc.z;
      grad->w = (m1.acc.w / (1.0 - powf(beta1, iter))) / (sqrtf((m2.acc.w / (1.0 - powf(beta2, iter)))) + eps) + weight_decay * weight.acc.w;

      weight_sum_sq += weight.acc.x * weight.acc.x + weight.acc.y * weight.acc.y + weight.acc.z * weight.acc.z + weight.acc.w * weight.acc.w;
      rtw_sum_sq += grad->x * grad->x + grad->y * grad->y + grad->z * grad->z + grad->w * grad->w;
    """
    )
    split_precomputation += """
    const auto weight_norm =
        sqrtf(GROUP_REDUCE_ALL_SUM(weight_sum_sq, at::acc_type<cache_t, true>));
    const auto rtw_norm =
        sqrtf(GROUP_REDUCE_ALL_SUM(rtw_sum_sq, at::acc_type<cache_t, true>));
     const auto true_ratio = weight_norm / rtw_norm;
  """
    split_weight_update = """
      weight_new.fma_(grad, -learning_rate * true_ratio);
    """
    split_weight_update_cpu = ""

    return {
        "optimizer": "lamb",
        "is_experimental_optimizer": True,
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "momentum2"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "beta1"),
                OptimItem(ArgType.FLOAT, "beta2"),
                OptimItem(ArgType.FLOAT, "weight_decay"),
                OptimItem(ArgType.INT, "iter"),
            ],
            {
                "v1": "Tensor momentum1, Tensor momentum2, float learning_rate = 0, float eps = 0, float beta1 = 0, float beta2 = 0, float weight_decay = 0, int iter = 0"
            },
        ),
        "split_precomputation": split_precomputation,
        "split_weight_update": split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": False,
        "has_gpu_support": True,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


def partial_rowwise_lamb() -> Dict[str, Any]:
    split_precomputation = """
    at::acc_type<cache_t, true> g_local_sum_square = 0.0;
    """
    split_precomputation += generate_optimized_grad_sum_loop_access(
        """
        const float4* grad = &{grad_vec}.acc;
        g_local_sum_square += grad->x * grad->x +
            grad->y * grad->y +
            grad->z * grad->z +
            grad->w * grad->w;
    """
    )
    split_precomputation += """
    const at::acc_type<cache_t, true> g_avg_square =
        GROUP_REDUCE_ALL_SUM(g_local_sum_square, at::acc_type<cache_t, true>) / D;

    at::acc_type<cache_t, true> m2;
    m2  = 0.0;
    if (threadIdx.x == 0) {
        m2 = beta2 * momentum2[idx] + (1.0 - beta2) * g_avg_square;
        momentum2[idx] = m2;
    }
    m2 = SHFL_SYNC(m2, 0);
    at::acc_type<cache_t, true> m2_hat = 1.0 / (sqrtf((m2 / (1.0 - powf(beta2, iter)))) + eps);

    at::acc_type<cache_t, true> weight_sum_sq = 0.0;
    at::acc_type<cache_t, true> rtw_sum_sq = 0.0;
    auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(weights, cache_weights, D);
    float2 qparams;
    if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
        qparams = weight_row.load_qparams();
    }
    """
    split_precomputation += generate_optimized_grad_sum_loop_access(
        """
        float4* grad = &{grad_vec}.acc;
        Vec4TAcc<cache_t> m1(&momentum1[idx * D + d]);
        m1.acc.x = beta1 * m1.acc.x + (1.0 - beta1) * grad->x;
        m1.acc.y = beta1 * m1.acc.y + (1.0 - beta1) * grad->y;
        m1.acc.z = beta1 * m1.acc.z + (1.0 - beta1) * grad->z;
        m1.acc.w = beta1 * m1.acc.w + (1.0 - beta1) * grad->w;
        m1.store(&momentum1[idx * D + d]);

        // now, we are finished with grad_sum. We can *reuse* grad_sum to store r_t + weight_decay * weight;
        Vec4TAcc<cache_t> weight = weight_row.load(d, qparams);
        grad->x = (m1.acc.x / (1.0 - powf(beta1, iter))) * m2_hat + weight_decay * weight.acc.x;
        grad->y = (m1.acc.y / (1.0 - powf(beta1, iter))) * m2_hat + weight_decay * weight.acc.y;
        grad->z = (m1.acc.z / (1.0 - powf(beta1, iter))) * m2_hat + weight_decay * weight.acc.z;
        grad->w = (m1.acc.w / (1.0 - powf(beta1, iter))) * m2_hat + weight_decay * weight.acc.w;

        weight_sum_sq += weight.acc.x * weight.acc.x + weight.acc.y * weight.acc.y + weight.acc.z * weight.acc.z + weight.acc.w * weight.acc.w;
        rtw_sum_sq += grad->x * grad->x + grad->y * grad->y + grad->z * grad->z + grad->w * grad->w;
    """
    )
    split_precomputation += """
    const auto weight_norm =
      sqrtf(GROUP_REDUCE_ALL_SUM(weight_sum_sq, at::acc_type<cache_t, true>));
    const auto rtw_norm =
      sqrtf(GROUP_REDUCE_ALL_SUM(rtw_sum_sq, at::acc_type<cache_t, true>));
    const auto true_ratio = weight_norm / rtw_norm;
    """

    split_weight_update = """
      weight_new.fma_(grad, -learning_rate * true_ratio);
    """
    split_weight_update_cpu = ""  # TODO

    return {
        "optimizer": "partial_rowwise_lamb",
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "momentum2"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "beta1"),
                OptimItem(ArgType.FLOAT, "beta2"),
                OptimItem(ArgType.FLOAT, "weight_decay"),
                OptimItem(ArgType.INT, "iter"),
            ],
            {
                "v1": "Tensor momentum1, Tensor momentum2, float learning_rate = 0, float eps = 0, float beta1 = 0, float beta2 = 0, float weight_decay = 0, int iter = 0"
            },
        ),
        "split_precomputation": split_precomputation,
        "split_weight_update": split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": False,
        "has_gpu_support": True,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


def adam() -> Dict[str, Any]:
    split_precomputation = """
    at::acc_type<cache_t, true>* __restrict__ row_counter;
    at::acc_type<cache_t, true> _row_counter = iter;
    if (use_rowwise_bias_correction) {
        const auto row_counter_placement = static_cast<PlacementType>(row_counter_placements[t]);
        const int64_t row_counter_offset = row_counter_offsets[t];
        if (row_counter_placement == PlacementType::DEVICE) {
            row_counter = &row_counter_dev[row_counter_offset];
        } else {
            row_counter = &row_counter_uvm[row_counter_offset];
        }

        // need to compute bias correction for each row
        if (threadIdx.x == 0) {
            _row_counter = row_counter[idx] + 1;
            row_counter[idx] = _row_counter;
        }

        // broadcast bias correction to all threads
        _row_counter = SHFL_SYNC(_row_counter, 0);
    }
    """

    split_weight_update = """
      Vec4T<cache_t> m_t(&momentum1[idx * D + d]);
      m_t.mul_(beta1);
      m_t.fma_(grad, 1.0 - beta1);
      m_t.store(&momentum1[idx * D + d]);

      Vec4T<cache_t> v_t(&momentum2[idx * D + d]);
      v_t.mul_(beta2);

      grad.acc.x *= grad.acc.x;
      grad.acc.y *= grad.acc.y;
      grad.acc.z *= grad.acc.z;
      grad.acc.w *= grad.acc.w;
      v_t.fma_(grad, 1.0 - beta2);
      v_t.store(&momentum2[idx * D + d]);

      weight_new.acc.x -= learning_rate * (m_t.acc.x / (1.0 - powf(beta1, _row_counter)) / (sqrtf((v_t.acc.x / (1.0 - powf(beta2, _row_counter)))) + eps) + weight_decay * weight_new.acc.x);
      weight_new.acc.y -= learning_rate * (m_t.acc.y / (1.0 - powf(beta1, _row_counter)) / (sqrtf((v_t.acc.y / (1.0 - powf(beta2, _row_counter)))) + eps) + weight_decay * weight_new.acc.y);
      weight_new.acc.z -= learning_rate * (m_t.acc.z / (1.0 - powf(beta1, _row_counter)) / (sqrtf((v_t.acc.z / (1.0 - powf(beta2, _row_counter)))) + eps) + weight_decay * weight_new.acc.z);
      weight_new.acc.w -= learning_rate * (m_t.acc.w / (1.0 - powf(beta1, _row_counter)) / (sqrtf((v_t.acc.w / (1.0 - powf(beta2, _row_counter)))) + eps) + weight_decay * weight_new.acc.w);
    """
    split_weight_update_cpu = ""  # TODO

    return {
        "optimizer": "adam",
        "is_experimental_optimizer": True,
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "momentum2"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "beta1"),
                OptimItem(ArgType.FLOAT, "beta2"),
                OptimItem(ArgType.FLOAT, "weight_decay"),
                OptimItem(ArgType.INT, "iter"),
                OptimItem(ArgType.BOOL, "use_rowwise_bias_correction"),
                OptimItem(ArgType.TENSOR, "row_counter", is_optional=True),
            ],
            {
                "v1": "Tensor momentum1, Tensor momentum2, float learning_rate = 0, float eps = 0, float beta1 = 0, float beta2 = 0, float weight_decay = 0, int iter = 0, bool use_rowwise_bias_correction = False, Tensor? row_counter = None",
            },
        ),
        "split_precomputation": split_precomputation,
        "split_weight_update": split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": False,
        "has_gpu_support": True,
        "has_vbe_support": True,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


def partial_rowwise_adam() -> Dict[str, Any]:
    split_precomputation = """
    at::acc_type<cache_t, true> g_local_sum_square = 0.0;
    """
    split_precomputation += generate_optimized_grad_sum_loop_access(
        """
        const float4* grad = &{grad_vec}.acc;
        g_local_sum_square += grad->x * grad->x +
            grad->y * grad->y +
            grad->z * grad->z +
            grad->w * grad->w;
    """
    )
    split_precomputation += """

    // Define the optimizer state (for use with optimizer offloading)
    struct OptimizerState {
        // momentum2 is a single value so it will be accessed directly as a struct field
        momentum2_ph_t momentum2;

        // momentum1 is an array of D values, so a method to return a pointer given the offset is defined instead
        DEVICE_INLINE momentum1_ph_t* momentum1_ptr() {
            // Re-cast the address to momentum1_ph_t* and return
            return reinterpret_cast<momentum1_ph_t *>(
                // Cast the address this to momentum2_t* and increment by 1 to skip over the momentum2 value
                reinterpret_cast<momentum2_ph_t *>(this) + 1
            );
        }
    };

    // Fetch the pointer to the optimizer state along the cache row
    [[maybe_unused]] auto* optimizer = weight_row_template.template optimizer_state_ptr<OptimizerState>();

    // Fetch the pointer to the momentum1 value
    // Define the fetch here instead of in split_weight_update to avoid conditionals inside a loop
    auto* momentum1_start = enable_optimizer_offloading ?
        (optimizer->momentum1_ptr()) :
        (&momentum1[idx * D]);

    const at::acc_type<cache_t, true> g_avg_square =
        GROUP_REDUCE_ALL_SUM(g_local_sum_square, at::acc_type<cache_t, true>) / D;

    at::acc_type<cache_t, true> v_hat_t;
    v_hat_t = 0.0;
    if (threadIdx.x == 0) {
        auto v_t = g_avg_square * (1.0 - beta2);

        if (enable_optimizer_offloading) {
            v_t += optimizer->momentum2 * beta2;
            optimizer->momentum2 = v_t;
        } else {
            v_t += momentum2[idx] * beta2;
            momentum2[idx] = v_t;
        }

        v_hat_t = v_t / (1.0 - powf(beta2, iter));
    }
    v_hat_t = SHFL_SYNC(v_hat_t, 0);
    """

    split_weight_update = """
      auto* momentum1_ptr = momentum1_start + d;
      Vec4T<momentum1_ph_t> m_t(momentum1_ptr);
      m_t.mul_(beta1);
      m_t.fma_(grad, 1.0 - beta1);
      m_t.store(momentum1_ptr);

      weight_new.acc.x -= learning_rate * (m_t.acc.x / (1.0 - powf(beta1, iter)) / (sqrtf(v_hat_t) + eps) + weight_decay * weight_new.acc.x);
      weight_new.acc.y -= learning_rate * (m_t.acc.y / (1.0 - powf(beta1, iter)) / (sqrtf(v_hat_t) + eps) + weight_decay * weight_new.acc.y);
      weight_new.acc.z -= learning_rate * (m_t.acc.z / (1.0 - powf(beta1, iter)) / (sqrtf(v_hat_t) + eps) + weight_decay * weight_new.acc.z);
      weight_new.acc.w -= learning_rate * (m_t.acc.w / (1.0 - powf(beta1, iter)) / (sqrtf(v_hat_t) + eps) + weight_decay * weight_new.acc.w);
    """
    split_weight_update_cpu = ""  # TODO

    return {
        "optimizer": "partial_rowwise_adam",
        "args": OptimizerArgsSet.create(
            [
                OptimItem(
                    ArgType.PLACEHOLDER_TENSOR,
                    "momentum1",
                    ph_tys=[ArgType.FLOAT_TENSOR, ArgType.BFLOAT16_TENSOR],
                ),
                OptimItem(
                    ArgType.PLACEHOLDER_TENSOR,
                    "momentum2",
                    ph_tys=[ArgType.FLOAT_TENSOR, ArgType.BFLOAT16_TENSOR],
                ),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eps"),
                OptimItem(ArgType.FLOAT, "beta1"),
                OptimItem(ArgType.FLOAT, "beta2"),
                OptimItem(ArgType.FLOAT, "weight_decay"),
                OptimItem(ArgType.INT, "iter"),
            ],
            {
                "v1": "Tensor momentum1, Tensor momentum2, float learning_rate = 0, float eps = 0, float beta1 = 0, float beta2 = 0, float weight_decay = 0, int iter = 0"
            },
        ),
        "split_precomputation": split_precomputation,
        "split_weight_update": split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": False,
        "has_gpu_support": True,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": True,
    }


def lars_sgd() -> Dict[str, Any]:
    split_precomputation = """
    at::acc_type<cache_t, true> weight_sum_sq = 0.0;
    at::acc_type<cache_t, true> grad_sum_sq = 0.0;

    auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(weights, cache_weights, D);
    float2 qparams;
    if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
        qparams = weight_row.load_qparams();
    }
    """
    split_precomputation += generate_optimized_grad_sum_loop_access(
        """
      const float4* grad = &{grad_vec}.acc;
      Vec4TAcc<cache_t> weight = weight_row.load(d, qparams);
      weight_sum_sq += weight.acc.x * weight.acc.x + weight.acc.y * weight.acc.y + weight.acc.z * weight.acc.z + weight.acc.w * weight.acc.w;
      grad_sum_sq += grad->x * grad->x + grad->y * grad->y + grad->z * grad->z + grad->w * grad->w;
    """
    )
    split_precomputation += """
    const auto weight_norm =
        sqrtf(GROUP_REDUCE_ALL_SUM(weight_sum_sq, at::acc_type<cache_t, true>));
    const auto grad_norm =
        sqrtf(GROUP_REDUCE_ALL_SUM(grad_sum_sq, at::acc_type<cache_t, true>));
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

    return {
        "optimizer": "lars_sgd",
        "is_experimental_optimizer": True,
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.TENSOR, "momentum1"),
                OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                OptimItem(ArgType.FLOAT, "eta"),
                OptimItem(ArgType.FLOAT, "momentum"),
                OptimItem(ArgType.FLOAT, "weight_decay"),
            ],
            {
                "v1": "Tensor momentum1, float learning_rate = 0, float eta = 0, float momentum = 0, float weight_decay = 0"
            },
        ),
        "split_precomputation": split_precomputation,
        "split_weight_update": split_weight_update,
        "split_post_update": "",
        "split_weight_update_cpu": split_weight_update_cpu,
        "has_cpu_support": False,
        "has_gpu_support": True,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }


def none_optimizer() -> Dict[str, Any]:
    return {
        "optimizer": "none",
        "dense": False,
        "args": OptimizerArgsSet.create(
            [
                OptimItem(ArgType.INT, "total_hash_size"),
                OptimItem(ArgType.SYM_INT, "total_unique_indices"),
            ],
            {
                "v1": "int total_hash_size = 0, SymInt total_unique_indices = 0",
            },
        ),
        # Generate only GPU code
        "has_cpu_support": False,
        "has_gpu_support": True,
        "has_vbe_support": False,
        "has_global_weight_decay_support": False,
        "has_ssd_support": False,
    }
