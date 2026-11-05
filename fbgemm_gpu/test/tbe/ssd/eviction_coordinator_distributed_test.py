#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import multiprocessing
import os
import tempfile
import traceback
import unittest
from collections.abc import Callable
from multiprocessing.queues import Queue

import torch
from fbgemm_gpu.tbe.ssd.eviction_coordinator import EvictionCoordinator
from torch import distributed as dist

_WORLD_SIZE = 2
_RESULT_TIMEOUT_S = 300


def _rank_main(
    func: Callable[[dist.ProcessGroup, int, int], object],
    rank: int,
    store_path: str,
    results: "Queue[tuple[int, bool, object]]",
) -> None:
    dist.init_process_group(
        backend="gloo",
        store=dist.FileStore(store_path, _WORLD_SIZE),
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        pg = dist.group.WORLD
        assert pg is not None
        results.put((rank, True, func(pg, rank, _WORLD_SIZE)))
    except Exception:
        results.put((rank, False, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def _run_on_gloo_ranks(
    func: Callable[[dist.ProcessGroup, int, int], object],
) -> list[object]:
    """Runs `func(pg, rank, world_size)` on two local Gloo ranks and returns the
    per-rank results in rank order."""
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        store_path = os.path.join(tmp, "store")
        procs = [
            ctx.Process(target=_rank_main, args=(func, rank, store_path, results))
            for rank in range(_WORLD_SIZE)
        ]
        for proc in procs:
            proc.start()
        by_rank: dict[int, object] = {}
        try:
            for _ in procs:
                rank, ok, value = results.get(timeout=_RESULT_TIMEOUT_S)
                if not ok:
                    raise AssertionError(f"rank {rank} failed:\n{value}")
                by_rank[rank] = value
        finally:
            for proc in procs:
                proc.join(timeout=_RESULT_TIMEOUT_S)
                if proc.is_alive():
                    proc.kill()
    return [by_rank[rank] for rank in range(_WORLD_SIZE)]


class _FakeBackend:
    def __init__(self, wants_eviction: bool, gate_open: bool) -> None:
        self.wants_eviction = wants_eviction
        self.gate_open = gate_open
        self.trigger_count = 0

    def should_trigger_evict(
        self, step: int, free_mem_bytes: int
    ) -> tuple[bool, bool, int]:
        return self.wants_eviction, self.gate_open, 0

    def trigger_feature_evict(self, force: bool = False) -> None:
        if force:
            self.trigger_count += 1

    def last_eviction_sufficient(self) -> bool:
        return True

    def restores_row_timestamps(self) -> bool:
        return False


class _FakeTBE:
    eviction_insufficient_backoff_batch: int = 1
    eviction_free_mem_floor_bytes: int = 0
    eviction_ttl_sweep_interval_s: float | None = None

    def __init__(
        self,
        name: str,
        wants_eviction: bool,
        gate_open: bool,
        automatic: bool = True,
        free_memory_trigger: bool = False,
        check_interval: int = 1,
    ) -> None:
        self.ssd_db = _FakeBackend(wants_eviction, gate_open)
        self.name = name
        self.automatic = automatic
        self.free_memory_trigger = free_memory_trigger
        self.eviction_check_interval_batch = check_interval

    def eviction_domain_key(self) -> str:
        return self.name

    def should_check_eviction(self, step: int) -> bool:
        return self.automatic and step % self.eviction_check_interval_batch == 0

    def uses_free_memory_trigger(self) -> bool:
        return self.free_memory_trigger

    def eviction_trigger_mode_id(self) -> int:
        return 5 if self.free_memory_trigger else 4


def _run_rank(
    pg: dist.ProcessGroup, rank: int, world_size: int
) -> tuple[int, int, int, int, int]:
    assert world_size == 2

    demand_coordinator = EvictionCoordinator()
    demand_tbe = _FakeTBE("demand", wants_eviction=rank == 0, gate_open=True)
    demand_coordinator.register(demand_tbe, pg)
    demand_coordinator.maybe_trigger(
        step=1,
        pg=pg,
        device=torch.device("cpu"),
        free_mem_bytes=0,
    )

    gate_coordinator = EvictionCoordinator()
    gate_tbe = _FakeTBE("gate", wants_eviction=True, gate_open=rank == 0)
    gate_coordinator.register(gate_tbe, pg)
    gate_coordinator.maybe_trigger(
        step=1,
        pg=pg,
        device=torch.device("cpu"),
        free_mem_bytes=0,
    )

    no_policy_coordinator = EvictionCoordinator()
    no_policy = _FakeTBE(
        "no_policy", wants_eviction=False, gate_open=True, automatic=False
    )
    enabled = _FakeTBE("enabled", wants_eviction=rank == 0, gate_open=True)
    no_policy_coordinator.register(no_policy, pg)
    no_policy_coordinator.register(enabled, pg)
    no_policy_coordinator.maybe_trigger(
        step=1,
        pg=pg,
        device=torch.device("cpu"),
        free_mem_bytes=0,
    )

    # A FREE_MEM table and an ID_COUNT table on the same pg: only rank 1's
    # ID_COUNT table is over budget. Both ranks must evict the ID_COUNT table
    # and neither may touch the FREE_MEM one.
    domain_coordinator = EvictionCoordinator()
    free_mem_tbe = _FakeTBE(
        "free_mem", wants_eviction=False, gate_open=True, free_memory_trigger=True
    )
    id_count_tbe = _FakeTBE("id_count", wants_eviction=rank == 1, gate_open=True)
    domain_coordinator.register(free_mem_tbe, pg)
    domain_coordinator.register(id_count_tbe, pg)
    domain_coordinator.maybe_trigger(
        step=1,
        pg=pg,
        device=torch.device("cpu"),
        free_mem_bytes=0,
    )

    return (
        demand_tbe.ssd_db.trigger_count,
        gate_tbe.ssd_db.trigger_count,
        enabled.ssd_db.trigger_count,
        free_mem_tbe.ssd_db.trigger_count,
        id_count_tbe.ssd_db.trigger_count,
    )


def _run_divergent_rank(pg: dist.ProcessGroup, rank: int, world_size: int) -> str:
    coordinator = EvictionCoordinator()
    # Rank 1 holds one more eviction-enabled TBE than rank 0. The registry holds
    # weak references, so keep the TBEs alive. With an interval above 1 nothing
    # is due at the retry step, so the registry slots must catch it there.
    tbes = [
        _FakeTBE(f"t{i}", wants_eviction=True, gate_open=True, check_interval=10)
        for i in range(rank + 1)
    ]
    for tbe in tbes:
        coordinator.register(tbe, pg)
    try:
        for step in (1, 2):
            coordinator.maybe_trigger(
                step=step,
                pg=pg,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )
    except RuntimeError as e:
        return "raised" if "layout differs across ranks" in str(e) else str(e)
    return "no error"


def _run_cadence_divergent_rank(
    pg: dist.ProcessGroup, rank: int, world_size: int
) -> tuple[str, int]:
    coordinator = EvictionCoordinator()
    # Both ranks register the same TBE, so the registries agree, but rank 0
    # checks it every 2 steps and rank 1 every 3. Without the agreed schedule,
    # each rank would enter collectives on its own steps.
    tbes = [
        _FakeTBE("t0", wants_eviction=False, gate_open=True, check_interval=2 + rank)
    ]
    for tbe in tbes:
        coordinator.register(tbe, pg)
    for step in range(1, 7):
        try:
            coordinator.maybe_trigger(
                step=step,
                pg=pg,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )
        except RuntimeError as e:
            if "layout differs across ranks" not in str(e):
                raise
            return "raised", step
    return "no error", 0


class EvictionCoordinatorDistributedTest(unittest.TestCase):
    def test_skewed_votes_are_globally_atomic(self) -> None:
        results = _run_on_gloo_ranks(_run_rank)
        self.assertEqual(results, [(1, 0, 1, 0, 1), (1, 0, 1, 0, 1)])

    def test_divergent_registries_fail_fast_on_every_rank(self) -> None:
        # The vote tensor has a fixed length, so the collective completes; the
        # mismatch survives the garbage collection and the retry at the next
        # step, and the header check raises on both ranks instead of hanging.
        results = _run_on_gloo_ranks(_run_divergent_rank)
        self.assertEqual(results, ["raised", "raised"])

    def test_divergent_cadences_fail_fast_before_a_lone_collective(self) -> None:
        # At step 1 the ranks agree on the registry but not on the next check
        # step (2 vs 3). Both retry at step 2, where they still disagree, so
        # both raise there instead of either calling a collective alone.
        results = _run_on_gloo_ranks(_run_cadence_divergent_rank)
        self.assertEqual(results, [("raised", 2), ("raised", 2)])


if __name__ == "__main__":
    unittest.main()
