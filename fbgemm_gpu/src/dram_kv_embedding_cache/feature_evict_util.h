// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace kv_mem {

template <typename weight_type>
std::unique_ptr<FeatureEvict<weight_type>> create_feature_evict(
    const FeatureEvictConfig& config,
    folly::CPUThreadPoolExecutor* executor,
    SynchronizedShardedMap<int64_t, weight_type*>& kv_store,
    const std::vector<int64_t>& sub_table_hash_cumsum) {
  if (executor == nullptr) {
    throw std::invalid_argument("executor cannot be null");
  }

  switch (config.trigger_strategy) {
    case EvictTriggerStrategy::BY_TIMESTAMP: {
      return std::make_unique<TimeBasedEvict<weight_type>>(
          executor, kv_store, sub_table_hash_cumsum, config.ttls_in_mins);
    }

    case EvictTriggerStrategy::BY_COUNTER: {
      for (auto count_decay_rate : config.counter_decay_rates) {
        if (count_decay_rate <= 0 || count_decay_rate > 1) {
          throw std::invalid_argument(
              "count_decay_rate must be in range (0,1]");
        }
      }
      return std::make_unique<CounterBasedEvict<weight_type>>(
          executor,
          kv_store,
          sub_table_hash_cumsum,
          config.counter_decay_rates,
          config.counter_thresholds);
    }

    case EvictTriggerStrategy::BY_TIMESTAMP_AND_COUNTER: {
      for (auto count_decay_rate : config.counter_decay_rates) {
        if (count_decay_rate <= 0 || count_decay_rate > 1) {
          throw std::invalid_argument(
              "count_decay_rate must be in range (0,1]");
        }
      }
      return std::make_unique<TimeCounterBasedEvict<weight_type>>(
          executor,
          kv_store,
          sub_table_hash_cumsum,
          config.ttls_in_mins,
          config.counter_decay_rates,
          config.counter_thresholds);
    }

    case EvictTriggerStrategy::BY_L2WEIGHT: {
      for (auto l2_weight_threshold : config.l2_weight_thresholds) {
        if (l2_weight_threshold < 0) {
          throw std::invalid_argument("l2_weight_threshold must be positive");
        }
      }
      for (auto embedding_dim : config.embedding_dims) {
        if (embedding_dim <= 0) {
          throw std::invalid_argument("embedding_dim must be positive");
        }
      }

      return std::make_unique<L2WeightBasedEvict<weight_type>>(
          executor,
          kv_store,
          sub_table_hash_cumsum,
          config.l2_weight_thresholds,
          config.embedding_dims);
    }

    default:
      throw std::runtime_error("Unknown evict trigger strategy");
  }
}

} // namespace kv_mem
