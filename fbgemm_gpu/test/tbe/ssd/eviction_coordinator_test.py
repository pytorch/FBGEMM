#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import contextlib
import itertools
import types
import unittest
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

import torch
from fbgemm_gpu.tbe.ssd.eviction_coordinator import (
    EvictionCoordinator,
    HEADER_SLOTS,
    MAX_DOMAINS,
    MAX_GATE_BLOCKED_CHECKS,
    SLOTS_PER_DOMAIN,
    VOTE_SLOTS,
)
from fbgemm_gpu.tbe.ssd.ssd_config import EvictionPolicy
from fbgemm_gpu.tbe.ssd.training import (
    DEFAULT_MEM_UTIL_EVICTION_CHECK_INTERVAL_BATCH,
    SSDTableBatchedEmbeddingBags,
)
from torch import distributed as dist

_GB: int = 1024**3


class _FakeBackend:
    def __init__(self, wants_eviction: bool, gate_open: bool, reading: int = 0) -> None:
        self.wants_eviction = wants_eviction
        self.gate_open = gate_open
        self.reading = reading
        self.sufficient = True
        self.restores_timestamps = False
        self.trigger_forces: list[bool] = []
        self.checked_inputs: list[tuple[int, int]] = []

    def should_trigger_evict(
        self, step: int, free_mem_bytes: int
    ) -> tuple[bool, bool, int]:
        self.checked_inputs.append((step, free_mem_bytes))
        return self.wants_eviction, self.gate_open, self.reading

    def trigger_feature_evict(self, force: bool = False) -> None:
        self.trigger_forces.append(force)

    def last_eviction_sufficient(self) -> bool:
        return self.sufficient

    def restores_row_timestamps(self) -> bool:
        return self.restores_timestamps


class _FakeTBE:
    _names: Iterator[int] = itertools.count()

    def __init__(
        self,
        wants_eviction: bool,
        gate_open: bool,
        check_interval: int = 1,
        automatic: bool = True,
        free_memory_trigger: bool = False,
        reading: int = 0,
        backoff: int = 1,
        floor: int = 0,
        name: str | None = None,
        ttl_sweep_interval_s: float | None = None,
    ) -> None:
        self.ssd_db = _FakeBackend(wants_eviction, gate_open, reading)
        self.check_interval = check_interval
        self.eviction_insufficient_backoff_batch = backoff
        self.eviction_free_mem_floor_bytes = floor
        self.eviction_ttl_sweep_interval_s = ttl_sweep_interval_s
        self.automatic = automatic
        self.free_memory_trigger = free_memory_trigger
        self.name: str = name or f"fake{next(_FakeTBE._names)}"

    def eviction_domain_key(self) -> str:
        return self.name

    @property
    def eviction_check_interval_batch(self) -> int:
        return self.check_interval

    def should_check_eviction(self, step: int) -> bool:
        return self.automatic and step % self.check_interval == 0

    def uses_free_memory_trigger(self) -> bool:
        return self.free_memory_trigger

    def eviction_trigger_mode_id(self) -> int:
        return 5 if self.free_memory_trigger else 4


def _reduce_as(
    world: int, *domains: tuple[int, int, int, int]
) -> Callable[[torch.Tensor, object, object], None]:
    """
    all_reduce stand-in: the header as if every rank had the same layout, and
    the given per-domain (wants, idle, sufficient, urgent) tallies.
    """

    def reduce(votes: torch.Tensor, op: object, group: object) -> None:
        reduced = votes.clone()
        reduced[:HEADER_SLOTS] = votes[:HEADER_SLOTS] * world
        for i, slots in enumerate(domains):
            base = HEADER_SLOTS + SLOTS_PER_DOMAIN * i
            reduced[base : base + SLOTS_PER_DOMAIN] = torch.tensor(
                slots, dtype=votes.dtype
            )
        votes.copy_(reduced)

    return reduce


@contextlib.contextmanager
def _distributed(
    world: int, reduce: Callable[[torch.Tensor, object, object], object]
) -> Iterator[MagicMock]:
    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=world),
        patch.object(dist, "get_backend", return_value=dist.Backend.GLOO),
        patch.object(dist, "all_reduce", side_effect=reduce) as all_reduce,
    ):
        yield all_reduce


def _check(
    coordinator: EvictionCoordinator, step: int, free_mem_bytes: int = 0
) -> None:
    coordinator.maybe_trigger(
        step=step,
        pg=None,
        device=torch.device("cpu"),
        free_mem_bytes=free_mem_bytes,
    )


class EvictionCoordinatorTest(unittest.TestCase):
    def test_step_zero_skips_cold_backend_coordination(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(tbe, None)

        with patch.object(dist, "all_reduce") as all_reduce:
            coordinator.maybe_trigger(
                step=0,
                pg=None,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )

        all_reduce.assert_not_called()
        self.assertEqual(tbe.ssd_db.checked_inputs, [])
        self.assertEqual(tbe.ssd_db.trigger_forces, [])

    def _log_for(
        self,
        coordinator: EvictionCoordinator,
        step: int,
        world: int,
        tally: tuple[int, int, int, int],
    ) -> str:
        with (
            _distributed(world, _reduce_as(world, tally)),
            self.assertLogs(
                "fbgemm_gpu.tbe.ssd.eviction_coordinator", level="INFO"
            ) as logs,
        ):
            _check(coordinator, step)
        self.assertEqual(len(logs.output), 1)
        return logs.output[0]

    def test_log_records_local_reading_votes_and_outcome(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True, reading=31402118)
        coordinator.register(tbe, None)

        message = self._log_for(coordinator, step=10, world=8, tally=(8, 8, 8, 0))

        self.assertIn("step=10", message)
        self.assertIn("mode=4", message)
        self.assertIn("local_read=31402118", message)
        self.assertIn("local_wants=1", message)
        self.assertIn("votes_wants=8/8", message)
        self.assertIn("votes_gate=8/8", message)
        self.assertIn("outcome=TRIGGERED", message)
        self.assertEqual(tbe.ssd_db.trigger_forces, [True])

    def test_non_scheduled_tbe_is_neither_triggered_nor_a_veto(self) -> None:
        coordinator = EvictionCoordinator()
        no_policy = _FakeTBE(wants_eviction=False, gate_open=False, automatic=False)
        enabled = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(no_policy, None)
        coordinator.register(enabled, None)

        with patch.object(dist, "is_initialized", return_value=False):
            coordinator.maybe_trigger(
                step=1,
                pg=None,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )

        # Its own domain is never due, so it is skipped outright: not consulted,
        # not triggered, and its closed gate cannot veto the sibling.
        self.assertEqual(no_policy.ssd_db.checked_inputs, [])
        self.assertEqual(no_policy.ssd_db.trigger_forces, [])
        self.assertEqual(enabled.ssd_db.trigger_forces, [True])

    def test_one_over_budget_tbe_does_not_evict_its_under_budget_peer(self) -> None:
        # The configuration that actually occurs: a model runs one trigger mode,
        # but its TBEs carry separate per-table row budgets and cross them at
        # different steps. compute_thresholds_from_buckets() evicts down to
        # keep_count, not trigger_count, so dragging the under-budget peer into
        # the round drops rows it was entitled to keep.
        coordinator = EvictionCoordinator()
        over_budget = _FakeTBE(wants_eviction=True, gate_open=True)
        under_budget = _FakeTBE(wants_eviction=False, gate_open=True)
        coordinator.register(over_budget, None)
        coordinator.register(under_budget, None)

        with patch.object(dist, "is_initialized", return_value=False):
            coordinator.maybe_trigger(
                step=1,
                pg=None,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )

        self.assertEqual(over_budget.ssd_db.trigger_forces, [True])
        self.assertEqual(under_budget.ssd_db.trigger_forces, [])

    def test_each_same_mode_tbe_gets_its_own_domain(self) -> None:
        coordinator = EvictionCoordinator()
        tbes = [_FakeTBE(wants_eviction=i == 1, gate_open=True) for i in range(3)]
        for tbe in tbes:
            coordinator.register(tbe, None)

        with patch.object(dist, "is_initialized", return_value=False):
            coordinator.maybe_trigger(
                step=1,
                pg=None,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )

        # Only the one that asked evicts; sharing a trigger mode does not merge
        # tables into a shared decision.
        self.assertEqual([tbe.ssd_db.trigger_forces for tbe in tbes], [[], [True], []])

    def test_free_mem_tables_share_a_single_domain(self) -> None:
        coordinator = EvictionCoordinator()
        quiet = _FakeTBE(wants_eviction=False, gate_open=True, free_memory_trigger=True)
        hungry = _FakeTBE(wants_eviction=True, gate_open=True, free_memory_trigger=True)
        coordinator.register(quiet, None)
        coordinator.register(hungry, None)

        with patch.object(dist, "is_initialized", return_value=False):
            coordinator.maybe_trigger(
                step=1,
                pg=None,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )

        # Free host memory is host-wide and cannot be attributed to a table, so
        # one table asking releases rows from every FREE_MEM table.
        self.assertEqual(quiet.ssd_db.trigger_forces, [True])
        self.assertEqual(hungry.ssd_db.trigger_forces, [True])

    def test_closed_gate_blocks_only_its_own_domain(self) -> None:
        coordinator = EvictionCoordinator()
        busy_free_mem = _FakeTBE(
            wants_eviction=True, gate_open=False, free_memory_trigger=True
        )
        id_count = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(busy_free_mem, None)
        coordinator.register(id_count, None)

        with patch.object(dist, "is_initialized", return_value=False):
            coordinator.maybe_trigger(
                step=1,
                pg=None,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )

        self.assertEqual(busy_free_mem.ssd_db.trigger_forces, [])
        self.assertEqual(id_count.ssd_db.trigger_forces, [True])

    def test_every_domain_votes_in_one_collective(self) -> None:
        coordinator = EvictionCoordinator()
        tbes = [_FakeTBE(wants_eviction=True, gate_open=True) for _ in range(3)]
        for tbe in tbes:
            coordinator.register(tbe, None)
        captured: list[list[int]] = []

        def reduce_unanimously(votes: torch.Tensor, op: object, group: object) -> None:
            captured.append(votes.tolist())
            votes.copy_(votes * 2)

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "get_backend", return_value=dist.Backend.GLOO),
            patch.object(dist, "all_reduce", side_effect=reduce_unanimously) as reduce,
        ):
            coordinator.maybe_trigger(
                step=1,
                pg=None,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )

        self.assertEqual(reduce.call_count, 1)
        # One (wants, idle, sufficient, urgent) slot group per domain, packed
        # after the header into a single fixed-length tensor.
        self.assertEqual(len(captured[0]), VOTE_SLOTS)
        self.assertEqual(captured[0][:2], [3, 9])
        domain_slots = captured[0][HEADER_SLOTS : HEADER_SLOTS + 3 * SLOTS_PER_DOMAIN]
        self.assertEqual(domain_slots, [1, 1, 1, 0] * 3)
        for tbe in tbes:
            self.assertEqual(tbe.ssd_db.trigger_forces, [True])

    def test_domain_slots_do_not_depend_on_registration_order(self) -> None:
        # Slots are ordered by a stable hash of each domain's mode and tables,
        # so two ranks agree on the layout even if their construction order
        # differed; a different set of domains still fails the header check.
        def capture_votes(order: list[int]) -> list[int]:
            coordinator = EvictionCoordinator()
            tbes = [
                _FakeTBE(wants_eviction=True, gate_open=True, name="asking"),
                _FakeTBE(
                    wants_eviction=False,
                    gate_open=True,
                    free_memory_trigger=True,
                    name="free_mem",
                ),
                _FakeTBE(wants_eviction=False, gate_open=True, name="quiet"),
            ]
            for i in order:
                coordinator.register(tbes[i], None)
            captured: list[list[int]] = []

            def capture(votes: torch.Tensor, op: object, group: object) -> None:
                captured.append(votes.tolist())
                votes.mul_(2)

            with _distributed(2, capture):
                _check(coordinator, step=1)
            return captured[0]

        self.assertEqual(capture_votes([0, 1, 2]), capture_votes([2, 0, 1]))

    def test_closed_gate_on_any_rank_suppresses_trigger(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(tbe, None)

        # The remote rank is still evicting; within the bounded wait, the round
        # waits for it.
        with _distributed(2, _reduce_as(2, (1, 1, 1, 0))) as all_reduce:
            _check(coordinator, step=1)

        self.assertEqual(all_reduce.call_count, 1)
        self.assertEqual(tbe.ssd_db.trigger_forces, [])

    def test_same_step_runs_exactly_one_collective_per_process_group(self) -> None:
        coordinator = EvictionCoordinator()
        first = _FakeTBE(wants_eviction=True, gate_open=True)
        second = _FakeTBE(wants_eviction=False, gate_open=True)
        coordinator.register(first, None)
        coordinator.register(second, None)

        with _distributed(2, lambda votes, op, group: votes.mul_(2)) as all_reduce:
            for _ in range(2):
                _check(coordinator, step=2)

        self.assertEqual(all_reduce.call_count, 1)
        self.assertEqual(first.ssd_db.checked_inputs, [(2, 0)])
        self.assertEqual(second.ssd_db.checked_inputs, [(2, 0)])

    def test_collectives_run_on_the_first_check_and_then_on_agreed_steps(
        self,
    ) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True, check_interval=10)
        coordinator.register(tbe, None)
        collective_steps: list[int] = []

        def record(votes: torch.Tensor, op: object, group: object) -> None:
            collective_steps.append(int(votes[4]))
            votes.mul_(2)

        with _distributed(2, record):
            for step in range(9, 22):
                _check(coordinator, step)

        # The first call votes with nothing due, to agree on step 10; after
        # that only the agreed steps run the collective.
        self.assertEqual(collective_steps, [9, 10, 20])
        self.assertEqual(tbe.ssd_db.checked_inputs, [(10, 0), (20, 0)])
        self.assertEqual(tbe.ssd_db.trigger_forces, [True, True])

    def test_rank_without_due_domains_still_votes_on_an_agreed_step(self) -> None:
        # Otherwise a rank whose registry diverged could skip a collective the
        # others enter, and they would hang in it.
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(
            wants_eviction=True, gate_open=True, automatic=False, check_interval=5
        )
        coordinator.register(tbe, None)
        headers: list[list[int]] = []

        def capture(votes: torch.Tensor, op: object, group: object) -> None:
            headers.append(votes[:HEADER_SLOTS].tolist())
            votes.mul_(2)

        with _distributed(2, capture):
            _check(coordinator, 3)
            _check(coordinator, 5)

        self.assertEqual(len(headers), 2)
        # No due domain, and the next agreed step.
        self.assertEqual(headers[1][:2], [0, 0])
        self.assertEqual(headers[1][6:8], [10, 100])
        # The registry is compared even with nothing due.
        self.assertEqual(headers[1][8:10], [1, 1])
        self.assertEqual(tbe.ssd_db.trigger_forces, [])

    def test_diverging_check_schedule_fails_fast(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True, check_interval=2)
        coordinator.register(tbe, None)

        def remote_checks_sooner(
            votes: torch.Tensor, op: object, group: object
        ) -> None:
            remote = votes.clone()
            remote[6] = 3
            remote[7] = 9
            votes.add_(remote)

        with _distributed(2, remote_checks_sooner):
            _check(coordinator, step=2)
            with self.assertRaisesRegex(RuntimeError, "layout differs across ranks"):
                _check(coordinator, step=3)
        self.assertEqual(tbe.ssd_db.trigger_forces, [])

    def test_free_memory_is_sampled_only_on_free_memory_cadence(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(
            wants_eviction=True,
            gate_open=True,
            check_interval=10,
            free_memory_trigger=True,
        )
        coordinator.register(tbe, None)
        samples = 0

        def sample_free_memory() -> int:
            nonlocal samples
            samples += 1
            return 123

        with patch.object(dist, "is_initialized", return_value=False):
            coordinator.maybe_trigger(
                step=9,
                pg=None,
                device=torch.device("cpu"),
                free_mem_bytes=sample_free_memory,
            )
            coordinator.maybe_trigger(
                step=10,
                pg=None,
                device=torch.device("cpu"),
                free_mem_bytes=sample_free_memory,
            )

        self.assertEqual(samples, 1)
        self.assertEqual(tbe.ssd_db.checked_inputs, [(10, 123)])
        self.assertEqual(tbe.ssd_db.trigger_forces, [True])

    def test_non_free_memory_mode_does_not_sample_free_memory(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(tbe, None)
        sample_free_memory = MagicMock(return_value=123)

        with patch.object(dist, "is_initialized", return_value=False):
            coordinator.maybe_trigger(
                step=1,
                pg=None,
                device=torch.device("cpu"),
                free_mem_bytes=sample_free_memory,
            )

        sample_free_memory.assert_not_called()
        self.assertEqual(tbe.ssd_db.checked_inputs, [(1, 0)])

    def test_process_groups_have_independent_registries_and_steps(self) -> None:
        coordinator = EvictionCoordinator()
        first_pg = MagicMock(spec=dist.ProcessGroup)
        second_pg = MagicMock(spec=dist.ProcessGroup)
        first = _FakeTBE(wants_eviction=True, gate_open=True)
        second = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(first, first_pg)
        coordinator.register(second, second_pg)

        with patch.object(dist, "is_initialized", return_value=False):
            coordinator.maybe_trigger(
                step=1,
                pg=first_pg,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )
            coordinator.maybe_trigger(
                step=1,
                pg=second_pg,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )

        self.assertEqual(first.ssd_db.trigger_forces, [True])
        self.assertEqual(second.ssd_db.trigger_forces, [True])
        self.assertEqual(first.ssd_db.checked_inputs, [(1, 0)])
        self.assertEqual(second.ssd_db.checked_inputs, [(1, 0)])

    def test_new_model_on_reused_group_does_not_inherit_stale_step(self) -> None:
        coordinator = EvictionCoordinator()
        pg = MagicMock(spec=dist.ProcessGroup)
        first = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(first, pg)

        with patch.object(dist, "is_initialized", return_value=False):
            coordinator.maybe_trigger(
                step=5,
                pg=pg,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )
            del first
            second = _FakeTBE(wants_eviction=True, gate_open=True)
            # Registering on a group that already checked resets it; the
            # registry emptying does not, since GC timing differs per rank.
            coordinator.register(second, pg)
            coordinator.maybe_trigger(
                step=1,
                pg=pg,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )

        self.assertEqual(second.ssd_db.trigger_forces, [True])

    def test_a_lagging_tbe_step_is_not_a_restart(self) -> None:
        # Under a pipelined trainer one TBE calls in a step behind another in
        # the same batch. Taking that for a restart would clear the backoff.
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(
            wants_eviction=True, gate_open=True, check_interval=10, backoff=30
        )
        tbe.ssd_db.sufficient = False
        coordinator.register(tbe, None)
        collective_steps: list[int] = []

        def record(votes: torch.Tensor, op: object, group: object) -> None:
            collective_steps.append(int(votes[4]))
            votes.mul_(2)

        with _distributed(2, record):
            for step in range(9, 72):
                _check(coordinator, step)
                _check(coordinator, step - 1)

        self.assertEqual(collective_steps, [9, 10, 20, 30, 40, 50, 60, 70])
        self.assertEqual(
            [step for step, _ in tbe.ssd_db.checked_inputs],
            [10, 20, 30, 40, 50, 60, 70],
        )
        # Rounds at 10, 40 and 70: the backoff held across the lagging calls.
        self.assertEqual(tbe.ssd_db.trigger_forces, [True, True, True])

    def test_registering_on_a_checked_group_resets_it(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(
            wants_eviction=True, gate_open=True, check_interval=10, backoff=30
        )
        tbe.ssd_db.sufficient = False
        coordinator.register(tbe, None)
        _check(coordinator, 10)
        _check(coordinator, 20)
        self.assertEqual(tbe.ssd_db.trigger_forces, [True])

        late = _FakeTBE(wants_eviction=False, gate_open=True, check_interval=10)
        coordinator.register(late, None)
        _check(coordinator, 30)

        # Without the reset, the backoff from step 10 would hold until 40.
        self.assertEqual(tbe.ssd_db.trigger_forces, [True, True])

    def test_a_large_step_drop_still_resets_the_group(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(
            wants_eviction=True, gate_open=True, check_interval=5, backoff=1000
        )
        tbe.ssd_db.sufficient = False
        coordinator.register(tbe, None)
        for step in (100, 105, 5):
            _check(coordinator, step)

        self.assertEqual([step for step, _ in tbe.ssd_db.checked_inputs], [100, 105, 5])
        self.assertEqual(tbe.ssd_db.trigger_forces, [True, True])

    def test_process_group_trigger_is_local_to_that_group(self) -> None:
        coordinator = EvictionCoordinator()
        first_pg = MagicMock(spec=dist.ProcessGroup)
        second_pg = MagicMock(spec=dist.ProcessGroup)
        first = _FakeTBE(wants_eviction=True, gate_open=True)
        second = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(first, first_pg)
        coordinator.register(second, second_pg)

        with patch.object(dist, "is_initialized", return_value=False):
            coordinator.maybe_trigger(
                step=1,
                pg=first_pg,
                device=torch.device("cpu"),
                free_mem_bytes=0,
            )

        self.assertEqual(first.ssd_db.trigger_forces, [True])
        self.assertEqual(second.ssd_db.trigger_forces, [])

    def test_distributed_vote_uses_cpu_for_gloo(self) -> None:
        coordinator = EvictionCoordinator()
        pg = MagicMock(spec=dist.ProcessGroup)
        tbe = _FakeTBE(wants_eviction=False, gate_open=True)
        coordinator.register(tbe, pg)
        # Only passed through; the collective is mocked, so no device is needed.
        accelerator = torch.device("cuda", 0)
        real_tensor = torch.tensor
        placements: list[tuple[torch.dtype, torch.device]] = []

        def make_vote(
            data: list[int], dtype: torch.dtype, device: torch.device
        ) -> torch.Tensor:
            placements.append((dtype, device))
            return real_tensor(data, dtype=dtype)

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "get_backend", return_value=dist.Backend.GLOO),
            patch.object(
                dist, "all_reduce", side_effect=lambda v, op, group: v.mul_(2)
            ),
            patch.object(torch, "tensor", side_effect=make_vote),
        ):
            coordinator.maybe_trigger(
                step=1,
                pg=pg,
                device=accelerator,
                free_mem_bytes=0,
            )

        self.assertEqual(placements, [(torch.int64, torch.device("cpu"))])

    def test_distributed_vote_uses_accelerator_for_nccl(self) -> None:
        coordinator = EvictionCoordinator()
        pg = MagicMock(spec=dist.ProcessGroup)
        tbe = _FakeTBE(wants_eviction=False, gate_open=True)
        coordinator.register(tbe, pg)
        # Only passed through; the collective is mocked, so no device is needed.
        accelerator = torch.device("cuda", 0)
        real_tensor = torch.tensor
        placements: list[tuple[torch.dtype, torch.device]] = []

        def make_vote(
            data: list[int], dtype: torch.dtype, device: torch.device
        ) -> torch.Tensor:
            placements.append((dtype, device))
            return real_tensor(data, dtype=dtype)

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "get_backend", return_value=dist.Backend.NCCL),
            patch.object(
                dist, "all_reduce", side_effect=lambda v, op, group: v.mul_(2)
            ),
            patch.object(torch, "tensor", side_effect=make_vote),
        ):
            coordinator.maybe_trigger(
                step=1,
                pg=pg,
                device=accelerator,
                free_mem_bytes=0,
            )

        self.assertEqual(placements, [(torch.int64, accelerator)])


class EvictionCheckCadenceTest(unittest.TestCase):
    """The per-mode cadence chosen by `_eviction_check_interval`."""

    @staticmethod
    def _interval(mode: int, **kwargs: object) -> int:
        # pyre-ignore[6]: kwargs are EvictionPolicy fields
        policy = EvictionPolicy(eviction_trigger_mode=mode, **kwargs)
        return SSDTableBatchedEmbeddingBags._eviction_check_interval(policy)

    def test_iteration_takes_its_cadence_from_the_step_interval(self) -> None:
        # Otherwise the frontend cadence and the backend predicate are two
        # independent gates and rounds land on the lcm of the two.
        self.assertEqual(self._interval(1, eviction_step_intervals=300), 300)

    def test_iteration_rejects_a_cadence_that_does_not_divide_the_interval(
        self,
    ) -> None:
        # 1000 against 300 would fire every 3000 steps, 3.3x too rarely.
        self.assertEqual(
            self._interval(
                1,
                eviction_step_intervals=300,
                eviction_check_interval_batch=1000,
            ),
            300,
        )

    def test_iteration_honours_a_cadence_that_divides_the_interval(self) -> None:
        # Checking every 100 steps and firing every 300 is coherent: the backend
        # predicate still gates, so the two agree and the caller gets what it
        # asked for rather than having it silently replaced by 300.
        self.assertEqual(
            self._interval(
                1,
                eviction_step_intervals=300,
                eviction_check_interval_batch=100,
            ),
            100,
        )

    def test_budgeted_modes_check_far_more_often_than_free_mem(self) -> None:
        # ID_COUNT compares against a target size with no slack, so the cadence
        # is the only thing bounding overshoot. FREE_MEM's threshold is headroom
        # and absorbs a coarse cadence by construction.
        id_count = self._interval(4)
        mem_util = self._interval(2)
        free_mem = self._interval(5)
        self.assertLess(id_count, mem_util)
        self.assertEqual(mem_util, DEFAULT_MEM_UTIL_EVICTION_CHECK_INTERVAL_BATCH)
        self.assertLess(mem_util, free_mem)

    def test_an_explicit_cadence_still_wins(self) -> None:
        self.assertEqual(self._interval(4, eviction_check_interval_batch=7), 7)
        self.assertEqual(self._interval(5, eviction_check_interval_batch=7), 7)

    def test_deprecated_free_mem_alias_applies_only_to_free_mem(self) -> None:
        self.assertEqual(self._interval(5, eviction_free_mem_check_interval_batch=7), 7)
        self.assertEqual(
            self._interval(
                5,
                eviction_free_mem_check_interval_batch=7,
                eviction_check_interval_batch=3,
            ),
            3,
        )
        # The old field was read only under free_mem, so it must not pin a
        # cadence for the other modes.
        self.assertEqual(
            self._interval(2, eviction_free_mem_check_interval_batch=7),
            DEFAULT_MEM_UTIL_EVICTION_CHECK_INTERVAL_BATCH,
        )


class InsufficientEvictionBackoffTest(unittest.TestCase):
    def _trigger_steps(self, sufficient: bool) -> list[int]:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True, backoff=30)
        tbe.ssd_db.sufficient = sufficient
        coordinator.register(tbe, None)
        triggered = []
        for step in range(10, 80, 10):
            before = len(tbe.ssd_db.trigger_forces)
            _check(coordinator, step)
            if len(tbe.ssd_db.trigger_forces) > before:
                triggered.append(step)
        return triggered

    def test_insufficient_round_blocks_until_the_backoff_elapses(self) -> None:
        self.assertEqual(self._trigger_steps(sufficient=False), [10, 40, 70])

    def test_sufficient_round_does_not_back_off(self) -> None:
        self.assertEqual(
            self._trigger_steps(sufficient=True), [10, 20, 30, 40, 50, 60, 70]
        )

    def test_any_sufficient_member_lifts_the_domain_backoff(self) -> None:
        coordinator = EvictionCoordinator()
        drained = _FakeTBE(
            wants_eviction=True, gate_open=True, free_memory_trigger=True, backoff=30
        )
        drained.ssd_db.sufficient = False
        busy = _FakeTBE(
            wants_eviction=False, gate_open=True, free_memory_trigger=True, backoff=30
        )
        coordinator.register(drained, None)
        coordinator.register(busy, None)

        _check(coordinator, 10)
        _check(coordinator, 20)

        self.assertEqual(drained.ssd_db.trigger_forces, [True, True])

    def _remote_backoff_steps(self, remote_sufficient: int) -> list[int]:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True, backoff=30)
        tbe.ssd_db.sufficient = False
        coordinator.register(tbe, None)
        triggered = []
        for step in (10, 20):
            before = len(tbe.ssd_db.trigger_forces)
            with _distributed(2, _reduce_as(2, (2, 2, remote_sufficient, 0))):
                _check(coordinator, step)
            if len(tbe.ssd_db.trigger_forces) > before:
                triggered.append(step)
        return triggered

    def test_productive_round_on_another_rank_lifts_the_backoff(self) -> None:
        # This rank evicted almost nothing, but the other one did: backing off
        # would let one quiet rank hold eviction off for the ranks that need it.
        self.assertEqual(self._remote_backoff_steps(remote_sufficient=1), [10, 20])

    def test_backoff_applies_when_no_rank_was_productive(self) -> None:
        self.assertEqual(self._remote_backoff_steps(remote_sufficient=0), [10])

    @staticmethod
    def _configured(policy: EvictionPolicy) -> types.SimpleNamespace:
        tbe = types.SimpleNamespace(
            _eviction_check_interval=SSDTableBatchedEmbeddingBags._eviction_check_interval,
            _eviction_free_mem_floor_bytes=SSDTableBatchedEmbeddingBags._eviction_free_mem_floor_bytes,
            _eviction_ttl_sweep_interval_s=SSDTableBatchedEmbeddingBags._eviction_ttl_sweep_interval_s,
        )
        SSDTableBatchedEmbeddingBags.set_eviction_trigger_config(
            tbe,  # pyre-ignore[6]
            policy,
        )
        return tbe

    def test_free_mem_default_backoff_is_two_check_intervals(self) -> None:
        # Ten FREE_MEM intervals would be 10,000 batches; two match the old 600 s
        # cooldown at ~2 steps/s.
        tbe = self._configured(
            EvictionPolicy(
                eviction_trigger_mode=5,
                eviction_free_mem_threshold_gb=100,
                eviction_check_interval_batch=1000,
            )
        )
        self.assertEqual(tbe.eviction_insufficient_backoff_batch, 2000)

    def test_default_backoff_is_ten_check_intervals(self) -> None:
        tbe = types.SimpleNamespace(
            _eviction_check_interval=SSDTableBatchedEmbeddingBags._eviction_check_interval,
            _eviction_free_mem_floor_bytes=SSDTableBatchedEmbeddingBags._eviction_free_mem_floor_bytes,
            _eviction_ttl_sweep_interval_s=SSDTableBatchedEmbeddingBags._eviction_ttl_sweep_interval_s,
        )
        SSDTableBatchedEmbeddingBags.set_eviction_trigger_config(
            tbe,  # pyre-ignore[6]
            EvictionPolicy(eviction_trigger_mode=4, eviction_check_interval_batch=7),
        )
        self.assertEqual(tbe.eviction_insufficient_backoff_batch, 70)

        SSDTableBatchedEmbeddingBags.set_eviction_trigger_config(
            tbe,  # pyre-ignore[6]
            EvictionPolicy(
                eviction_trigger_mode=4,
                eviction_check_interval_batch=7,
                eviction_insufficient_backoff_batch=15,
            ),
        )
        self.assertEqual(tbe.eviction_insufficient_backoff_batch, 15)

    def test_backoff_not_above_the_check_interval_warns(self) -> None:
        # The next check is one interval later, past such a backoff.
        with self.assertLogs(level="WARNING"):
            self._configured(
                EvictionPolicy(
                    eviction_trigger_mode=4,
                    eviction_check_interval_batch=7,
                    eviction_insufficient_backoff_batch=7,
                )
            )


class VoteLayoutTest(unittest.TestCase):
    @staticmethod
    def _vote_length(domain_count: int) -> int:
        coordinator = EvictionCoordinator()
        # The registry holds weak references, so keep the TBEs alive.
        tbes = [
            _FakeTBE(wants_eviction=False, gate_open=True) for _ in range(domain_count)
        ]
        for tbe in tbes:
            coordinator.register(tbe, None)
        captured: list[int] = []

        def capture(votes: torch.Tensor, op: object, group: object) -> None:
            captured.append(votes.numel())
            votes.mul_(2)

        with _distributed(2, capture):
            _check(coordinator, step=1)
        return captured[0]

    def test_vote_length_does_not_depend_on_due_domains(self) -> None:
        # Ranks whose registries diverge must still hand the collective
        # equal-length tensors, so they fail the header check instead of
        # hanging it.
        self.assertEqual(self._vote_length(1), VOTE_SLOTS)
        self.assertEqual(self._vote_length(3), VOTE_SLOTS)

    @staticmethod
    def _remote_has_another_domain(
        votes: torch.Tensor, op: object, group: object
    ) -> None:
        remote = votes.clone()
        remote[0] += 1
        remote[1] = remote[0] * remote[0]
        votes.add_(remote)

    def test_layout_mismatch_that_survives_collection_fails_fast(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(tbe, None)

        with _distributed(2, self._remote_has_another_domain):
            with patch("gc.collect") as collect:
                _check(coordinator, step=1)
            collect.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "layout differs across ranks"):
                _check(coordinator, step=2)
        self.assertEqual(tbe.ssd_db.trigger_forces, [])

    def test_registry_mismatch_persisting_with_nothing_due_raises_at_retry(
        self,
    ) -> None:
        # With a check interval above 1 nothing is due at the retry step, so
        # only the registry slots can show that the divergence persists.
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True, check_interval=10)
        coordinator.register(tbe, None)

        def remote_registers_another_domain(
            votes: torch.Tensor, op: object, group: object
        ) -> None:
            remote = votes.clone()
            remote[8] += 1
            remote[9] = remote[8] * remote[8]
            votes.add_(remote)

        with _distributed(2, remote_registers_another_domain), patch("gc.collect"):
            _check(coordinator, step=10)
            with self.assertRaisesRegex(RuntimeError, "layout differs across ranks"):
                _check(coordinator, step=11)
        self.assertEqual(tbe.ssd_db.trigger_forces, [])

    def test_layout_mismatch_cleared_by_collection_votes_again_next_step(
        self,
    ) -> None:
        # A rebuilt model's old TBE can stay registered on one rank until its
        # garbage collector runs; the collection makes the layouts agree.
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True, check_interval=10)
        coordinator.register(tbe, None)

        with patch("gc.collect"):
            with _distributed(2, self._remote_has_another_domain):
                _check(coordinator, step=10)
            with _distributed(2, _reduce_as(2, (2, 2, 2, 0))):
                _check(coordinator, step=11)
                _check(coordinator, step=20)

        # Step 11 is a collective with nothing due; the round runs at step 20.
        self.assertEqual(tbe.ssd_db.trigger_forces, [True])

    def test_too_many_domains_is_rejected_at_registration(self) -> None:
        coordinator = EvictionCoordinator()
        tbes = [
            _FakeTBE(wants_eviction=False, gate_open=True)
            for _ in range(MAX_DOMAINS + 1)
        ]
        for tbe in tbes[:-1]:
            coordinator.register(tbe, None)

        with self.assertRaises(ValueError):
            coordinator.register(tbes[-1], None)


class BoundedGateWaitTest(unittest.TestCase):
    def test_round_starts_after_bounded_wait_on_a_busy_rank(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(tbe, None)
        forced_at = []
        for step in range(1, MAX_GATE_BLOCKED_CHECKS + 2):
            before = len(tbe.ssd_db.trigger_forces)
            with _distributed(2, _reduce_as(2, (1, 1, 1, 0))):
                _check(coordinator, step)
            if len(tbe.ssd_db.trigger_forces) > before:
                forced_at.append(step)

        # Waits MAX_GATE_BLOCKED_CHECKS - 1 checks for the busy rank, then
        # starts the round; the counter restarts after it.
        self.assertEqual(forced_at, [MAX_GATE_BLOCKED_CHECKS])

    def test_demand_going_away_resets_the_wait(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=True, gate_open=True)
        coordinator.register(tbe, None)
        tallies = [(1, 1, 1, 0)] * (MAX_GATE_BLOCKED_CHECKS - 1) + [
            (0, 2, 1, 0),
            (1, 1, 1, 0),
        ]
        for step, tally in enumerate(tallies, start=1):
            with _distributed(2, _reduce_as(2, tally)):
                _check(coordinator, step)

        self.assertEqual(tbe.ssd_db.trigger_forces, [])


class FreeMemoryFloorTest(unittest.TestCase):
    def test_low_free_memory_starts_a_round_without_demand(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=False, gate_open=True, floor=100)
        coordinator.register(tbe, None)

        _check(coordinator, step=1, free_mem_bytes=99)

        self.assertEqual(tbe.ssd_db.trigger_forces, [True])

    def _urgent_steps(self, sufficient: bool) -> tuple[list[int], list[str]]:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=False, gate_open=True, backoff=30, floor=100)
        tbe.ssd_db.sufficient = sufficient
        coordinator.register(tbe, None)
        triggered = []
        with self.assertLogs(
            "fbgemm_gpu.tbe.ssd.eviction_coordinator", level="INFO"
        ) as logs:
            for step in range(10, 80, 10):
                before = len(tbe.ssd_db.trigger_forces)
                _check(coordinator, step, free_mem_bytes=99)
                if len(tbe.ssd_db.trigger_forces) > before:
                    triggered.append(step)
        warnings = [line for line in logs.output if line.startswith("WARNING")]
        return triggered, warnings

    def test_urgent_round_that_evicts_nothing_backs_off_and_is_logged(self) -> None:
        # Re-running a round that just found nothing to evict would find
        # nothing again, every check.
        triggered, warnings = self._urgent_steps(sufficient=False)

        self.assertEqual(triggered, [10, 40, 70])
        # Once per backoff window, not once per check.
        self.assertEqual(len(warnings), 2)
        self.assertIn("next urgent round waits until step 40", warnings[0])

    def test_urgent_rounds_repeat_while_they_evict(self) -> None:
        triggered, warnings = self._urgent_steps(sufficient=True)

        self.assertEqual(triggered, [10, 20, 30, 40, 50, 60, 70])
        self.assertEqual(warnings, [])

    def test_low_free_memory_on_another_rank_bypasses_a_busy_rank(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=False, gate_open=True, floor=100)
        coordinator.register(tbe, None)

        with _distributed(2, _reduce_as(2, (0, 1, 1, 1))):
            _check(coordinator, step=1, free_mem_bytes=1000)

        self.assertEqual(tbe.ssd_db.trigger_forces, [True])

    def test_zero_floor_disables_it(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=False, gate_open=True, floor=0)
        coordinator.register(tbe, None)
        sample_free_memory = MagicMock(return_value=0)

        coordinator.maybe_trigger(
            step=1,
            pg=None,
            device=torch.device("cpu"),
            free_mem_bytes=sample_free_memory,
        )

        sample_free_memory.assert_not_called()
        self.assertEqual(tbe.ssd_db.trigger_forces, [])

    def test_floor_samples_free_memory_in_budgeted_modes(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=False, gate_open=True, floor=100)
        coordinator.register(tbe, None)

        coordinator.maybe_trigger(
            step=1,
            pg=None,
            device=torch.device("cpu"),
            free_mem_bytes=MagicMock(return_value=99),
        )

        self.assertEqual(tbe.ssd_db.checked_inputs, [(1, 99)])
        self.assertEqual(tbe.ssd_db.trigger_forces, [True])

    def test_default_floor_is_off_without_meminfo(self) -> None:
        floor = SSDTableBatchedEmbeddingBags._eviction_free_mem_floor_bytes
        with patch("sys.platform", "darwin"):
            self.assertEqual(floor(EvictionPolicy(eviction_trigger_mode=4)), 0)
            self.assertEqual(
                floor(
                    EvictionPolicy(
                        eviction_trigger_mode=4, eviction_free_mem_floor_gb=7
                    )
                ),
                7 * _GB,
            )

    def test_floor_resolution(self) -> None:
        floor = SSDTableBatchedEmbeddingBags._eviction_free_mem_floor_bytes
        with patch.object(
            SSDTableBatchedEmbeddingBags,
            "get_total_cpu_memory_bytes",
            return_value=1000 * _GB,
        ):
            # Outside FREE_MEM the floor is only a backstop: 2% of MemTotal.
            self.assertEqual(floor(EvictionPolicy(eviction_trigger_mode=4)), 20 * _GB)
            self.assertEqual(
                floor(
                    EvictionPolicy(
                        eviction_trigger_mode=4, eviction_free_mem_floor_gb=7
                    )
                ),
                7 * _GB,
            )
            self.assertEqual(
                floor(
                    EvictionPolicy(
                        eviction_trigger_mode=4, eviction_free_mem_floor_gb=0
                    )
                ),
                0,
            )
            self.assertEqual(floor(EvictionPolicy(eviction_trigger_mode=3)), 0)
            # FREE_MEM keeps the floor below its own threshold, so its rounds
            # do not all become urgent and skip the backoff.
            self.assertEqual(
                floor(
                    EvictionPolicy(
                        eviction_trigger_mode=5, eviction_free_mem_threshold_gb=60
                    )
                ),
                30 * _GB,
            )
            self.assertEqual(
                floor(
                    EvictionPolicy(
                        eviction_trigger_mode=5, eviction_free_mem_threshold_gb=400
                    )
                ),
                50 * _GB,
            )


class TtlSweepTest(unittest.TestCase):
    def _sweep_times(
        self, times: list[float], restores_timestamps: bool = False
    ) -> list[float]:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=False, gate_open=True, ttl_sweep_interval_s=60)
        tbe.ssd_db.restores_timestamps = restores_timestamps
        coordinator.register(tbe, None)
        swept = []
        for step, now in enumerate(times, start=1):
            before = len(tbe.ssd_db.trigger_forces)
            with patch("time.monotonic", return_value=now):
                _check(coordinator, step)
            if len(tbe.ssd_db.trigger_forces) > before:
                swept.append(now)
        return swept

    def test_ttl_sweep_runs_once_per_interval_without_demand(self) -> None:
        # The first check starts the clock, and each sweep restarts it.
        self.assertEqual(
            self._sweep_times([0.0, 59.0, 60.0, 100.0, 119.0, 120.0]),
            [60.0, 120.0],
        )

    def test_restarted_job_sweeps_on_its_first_check_if_timestamps_restored(
        self,
    ) -> None:
        # Restored rows may already be past their TTL, and a job preempted more
        # often than the interval must still sweep them.
        for _ in range(3):
            self.assertEqual(
                self._sweep_times([0.0, 59.0], restores_timestamps=True), [0.0]
            )

    def test_restarted_job_starts_the_clock_if_restore_restamps(self) -> None:
        # Every row is younger than the process, so the first check sweeps
        # nothing and only starts the clock.
        for _ in range(3):
            self.assertEqual(self._sweep_times([0.0, 59.0]), [])
        self.assertEqual(self._sweep_times([0.0, 59.0, 60.0]), [60.0])

    def test_tbe_without_ttl_sweep_never_sweeps(self) -> None:
        coordinator = EvictionCoordinator()
        tbe = _FakeTBE(wants_eviction=False, gate_open=True)
        coordinator.register(tbe, None)
        for step, now in enumerate([0.0, 1e6], start=1):
            with patch("time.monotonic", return_value=now):
                _check(coordinator, step)

        self.assertEqual(tbe.ssd_db.trigger_forces, [])

    def test_sweep_interval_resolution(self) -> None:
        interval = SSDTableBatchedEmbeddingBags._eviction_ttl_sweep_interval_s

        def policy(mode: int, ttls: list[int]) -> EvictionPolicy:
            return EvictionPolicy(
                eviction_trigger_mode=mode,
                eviction_strategy=5,
                eviction_free_mem_threshold_gb=100,
                ttls_in_mins=ttls,
                training_id_eviction_trigger_count=[10] * len(ttls),
                training_id_keep_count=[5] * len(ttls),
                feature_score_counter_decay_rates=[0.99] * len(ttls),
                enable_eviction_for_feature_score_eviction_policy=[True] * len(ttls),
            )

        # Shortest TTL 60 min, swept 10 times per TTL.
        self.assertEqual(interval(policy(4, [0, 60, 1440])), 360.0)
        # Summed ID_COUNT already counts TTL rows; FREE_MEM is host-wide.
        self.assertIsNone(interval(policy(4, [60, 1440])))
        self.assertIsNone(interval(policy(4, [0, 0])))
        self.assertIsNone(interval(policy(5, [0, 60])))


if __name__ == "__main__":
    unittest.main()
