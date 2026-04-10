# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[29]

import tempfile
import unittest
from typing import Optional

from fbgemm_gpu.runtime_monitor import TBEStatsReporter, TBEStatsReporterConfig
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BackendType,
    BoundsCheckMode,
    PoolingMode,
)
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags

from .. import common  # noqa E402
from ..common import gpu_unavailable, running_in_oss


class CrashingStatsReporter(TBEStatsReporter):
    """Reporter that crashes on report to test exception isolation."""

    def __init__(self, report_interval: int = 100) -> None:
        self.report_interval: int = report_interval

    def should_report(self, iteration_step: int) -> bool:
        return True

    def register_stats(self, stats_name: str, amplifier: int = 1) -> None:
        pass

    def report_duration(
        self,
        iteration_step: int,
        event_name: str,
        duration_ms: float,
        embedding_id: str = "",
        tbe_id: str = "",
        time_unit: str = "ms",
        enable_tb_metrics: bool = False,
    ) -> None:
        raise RuntimeError("Simulated reporter crash")

    def report_data_amount(
        self,
        iteration_step: int,
        event_name: str,
        data_bytes: int,
        embedding_id: str = "",
        tbe_id: str = "",
        enable_tb_metrics: bool = False,
    ) -> None:
        raise RuntimeError("Simulated reporter crash")


class MockStatsReporter(TBEStatsReporter):
    """A simple mock stats reporter for testing."""

    def __init__(self, report_interval: int) -> None:
        self.report_interval: int = report_interval

    def should_report(self, iteration_step: int) -> bool:
        return iteration_step > 0 and iteration_step % self.report_interval == 0

    def register_stats(self, stats_name: str, amplifier: int = 1) -> None:
        pass

    def report_duration(
        self,
        iteration_step: int,
        event_name: str,
        duration_ms: float,
        embedding_id: str = "",
        tbe_id: str = "",
        time_unit: str = "ms",
        enable_tb_metrics: bool = False,
    ) -> None:
        pass

    def report_data_amount(
        self,
        iteration_step: int,
        event_name: str,
        data_bytes: int,
        embedding_id: str = "",
        tbe_id: str = "",
        enable_tb_metrics: bool = False,
    ) -> None:
        pass


class MockStatsReporterConfig(TBEStatsReporterConfig):
    """Config that creates a MockStatsReporter and stores it for test access."""

    def __init__(self, interval: int) -> None:
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "_reporter", None)

    def create_reporter(self) -> Optional[TBEStatsReporter]:
        if self.interval <= 0:
            return None
        reporter = MockStatsReporter(report_interval=self.interval)
        object.__setattr__(self, "_reporter", reporter)
        return reporter


def create_test_ssd_tbe(
    stats_reporter_config: Optional[TBEStatsReporterConfig] = None,
) -> SSDTableBatchedEmbeddingBags:
    """Create a minimal SSD TBE for alerting tests."""
    T = 2
    D = 8
    E = 100
    embedding_specs = [(E, D)] * T
    tmpdir = tempfile.mkdtemp()
    emb = SSDTableBatchedEmbeddingBags(
        embedding_specs=embedding_specs,
        feature_table_map=list(range(T)),
        cache_sets=10,
        ssd_storage_directory=tmpdir,
        ssd_rocksdb_shards=1,
        weights_precision=SparseType.FP32,
        output_dtype=SparseType.FP32,
        optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        pooling_mode=PoolingMode.SUM,
        bounds_check_mode=BoundsCheckMode.NONE,
        use_passed_in_path=True,
        gather_ssd_cache_stats=True,
        stats_reporter_config=stats_reporter_config,
        backend_type=BackendType.SSD,
    )
    return emb


@unittest.skipIf(*running_in_oss)
@unittest.skipIf(*gpu_unavailable)
class SSDAlertingTest(unittest.TestCase):
    """Tests for SSD TBE failure logging and alerting."""

    def test_reporter_exception_does_not_crash(self) -> None:
        """Verify stats reporter exceptions are caught and don't crash training."""
        config = MockStatsReporterConfig(interval=10)
        emb = create_test_ssd_tbe(stats_reporter_config=config)
        # Replace reporter with crashing one
        emb.stats_reporter = CrashingStatsReporter()
        emb.step = 10
        emb.last_reported_step = 0
        emb.last_reported_ssd_stats = [0.0] * emb.ssd_cache_stats_size
        # These should NOT raise -- the try/except should catch
        emb._report_ssd_l1_cache_stats()
        emb._report_ssd_io_stats()
        emb._report_ssd_mem_usage()
        emb._report_l2_cache_perf_stats()
        # If we get here without exception, the test passes

    def test_reporter_exception_logged_with_prefix(self) -> None:
        """Verify caught reporter exceptions use [SSD Offloading] prefix."""
        config = MockStatsReporterConfig(interval=10)
        emb = create_test_ssd_tbe(stats_reporter_config=config)
        emb.stats_reporter = CrashingStatsReporter()
        emb.step = 10
        emb.last_reported_step = 0
        emb.last_reported_ssd_stats = [0.0] * emb.ssd_cache_stats_size
        with self.assertLogs(level="WARNING") as cm:
            emb._report_ssd_l1_cache_stats()
        self.assertTrue(
            any("[SSD Offloading]" in msg for msg in cm.output),
            f"Expected '[SSD Offloading]' prefix in log output: {cm.output}",
        )

    def test_disk_space_check_runs(self) -> None:
        """Verify disk space check executes without crash."""
        config = MockStatsReporterConfig(interval=10)
        emb = create_test_ssd_tbe(stats_reporter_config=config)
        emb.step = 10
        emb.last_reported_step = 0
        emb.last_reported_ssd_stats = [0.0] * emb.ssd_cache_stats_size
        # Should run without crash, including disk space check
        emb._report_kv_backend_stats()

    def test_flush_logging(self) -> None:
        """Verify flush includes [SSD Offloading] [Rank N] logging."""
        config = MockStatsReporterConfig(interval=10)
        emb = create_test_ssd_tbe(stats_reporter_config=config)
        with self.assertLogs(level="INFO") as cm:
            emb.flush()
        self.assertTrue(
            any("[SSD Offloading]" in msg and "[Rank" in msg for msg in cm.output),
            f"Expected '[SSD Offloading] [Rank N]' in flush logs: {cm.output}",
        )

    def test_lazy_init_error_propagated(self) -> None:
        """Verify _lazy_init_error raises RuntimeError on ssd_db access."""
        config = MockStatsReporterConfig(interval=10)
        emb = create_test_ssd_tbe(stats_reporter_config=config)
        # Force access to ssd_db first to complete lazy init
        _ = emb.ssd_db
        # Now simulate a background init failure
        emb._lazy_init_error = RuntimeError("Simulated init crash")
        # Accessing ssd_db should now raise because _lazy_init_error is set
        # (after the impl fix, the error check is outside the thread-join block)
        with self.assertRaises(RuntimeError) as ctx:
            _ = emb.ssd_db
        self.assertIn("[SSD Offloading]", str(ctx.exception))
        self.assertIn("Simulated init crash", str(ctx.exception))

    def test_all_report_methods_handle_exceptions(self) -> None:
        """Verify all _report_* methods catch exceptions from crashing reporter."""
        config = MockStatsReporterConfig(interval=10)
        emb = create_test_ssd_tbe(stats_reporter_config=config)
        # Replace with crashing reporter to test exception isolation
        emb.stats_reporter = CrashingStatsReporter()
        emb.step = 10
        emb.last_reported_step = 0
        emb.last_reported_ssd_stats = [0.0] * emb.ssd_cache_stats_size
        # All individual report methods should catch and not crash
        emb._report_ssd_l1_cache_stats()
        emb._report_ssd_io_stats()
        emb._report_ssd_mem_usage()
        emb._report_l2_cache_perf_stats()
        # Full orchestrator should also not crash
        emb._report_kv_backend_stats()
