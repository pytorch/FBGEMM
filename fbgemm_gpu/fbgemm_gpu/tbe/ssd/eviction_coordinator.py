#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Rank-synchronized eviction triggering for KVZCH TBEs."""

import gc
import hashlib
import itertools
import logging
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import distributed as dist

logger: logging.Logger = logging.getLogger(__name__)

# ITERATION, MEM_UTIL, ID_COUNT, FREE_MEM. DISABLED (0) and MANUAL (3) never
# self-trigger, so they are not registered: they must neither vote in a
# readiness gate nor be force-evicted by another table's decision.
AUTOMATIC_TRIGGER_MODES: frozenset[int] = frozenset({1, 2, 4, 5})

# Domain key shared by every FREE_MEM table on the rank.
_FREE_MEM_DOMAIN = "free_mem"

# The vote tensor has a fixed length, so ranks whose registries diverge still
# pass equal-length tensors to the collective and fail the header check rather
# than hanging it.
MAX_DOMAINS = 64
# For the due domains: n, n^2, layout, layout^2; step, step^2; next check
# step, its square; for every registered domain: count, count^2, layout,
# layout^2. The registry slots make a check with nothing due still compare the
# registries.
HEADER_SLOTS = 12
# wants, idle, sufficient, urgent
SLOTS_PER_DOMAIN = 4
VOTE_SLOTS: int = HEADER_SLOTS + SLOTS_PER_DOMAIN * MAX_DOMAINS
# Header values are reduced modulo this prime so that x^2 summed over any
# realistic world size stays inside int64.
_HEADER_MODULUS = 1048573

# Consecutive checks a domain may want a round while some rank is still busy
# before the round starts anyway. A rank still evicting ignores the trigger.
MAX_GATE_BLOCKED_CHECKS = 3

# Every registered TBE calls in with its own step counter, and under a
# pipelined trainer those counters lag one another by a step or two. A step at
# most this far below the last one seen is such a lagging caller, for a batch
# that was already checked, not a restart of the step count.
_MAX_STEP_LAG = 8


class _EvictionBackend(Protocol):
    # Docstring bodies rather than `...`: OSS flake8 runs without E704 ignored.
    def should_trigger_evict(
        self, step: int, free_mem_bytes: int
    ) -> tuple[bool, bool, int]:
        """(wants a round, no round in flight, the reading the mode judged)."""

    def trigger_feature_evict(self, force: bool = False) -> None:
        """Start a round; `force` skips the wall-clock interval gate."""

    def last_eviction_sufficient(self) -> bool:
        """Whether the last finished round evicted more than 100 rows."""

    def restores_row_timestamps(self) -> bool:
        """Whether a checkpoint restore keeps the saved row timestamps."""


class EvictionParticipant(Protocol):
    @property
    def ssd_db(self) -> _EvictionBackend:
        """The backend that votes and evicts."""

    def should_check_eviction(self, step: int) -> bool:
        """Whether this TBE is in training and due for a check at `step`."""

    def uses_free_memory_trigger(self) -> bool:
        """Whether this TBE's trigger mode is FREE_MEM."""

    def eviction_trigger_mode_id(self) -> int:
        """The trigger mode, for logs."""

    def eviction_domain_key(self) -> str:
        """Trigger mode and table names, identical on every rank."""

    @property
    def eviction_check_interval_batch(self) -> int:
        """Batches between checks."""

    @property
    def eviction_insufficient_backoff_batch(self) -> int:
        """Batches to wait after a round that no rank found sufficient."""

    @property
    def eviction_free_mem_floor_bytes(self) -> int:
        """Host free memory below which this rank votes urgent; 0 disables it."""

    @property
    def eviction_ttl_sweep_interval_s(self) -> float | None:
        """Seconds between TTL sweeps, or None if this TBE needs none."""


@dataclass
class _Domain:
    id: int
    label: str
    tbes: list[EvictionParticipant]


@dataclass
class _Tally:
    wants: int
    idle: int
    sufficient: int
    urgent: int


def _stable_id(key: str) -> int:
    # Not hash(): str hashing is salted per process, so ranks would disagree.
    return int.from_bytes(
        hashlib.blake2b(key.encode(), digest_size=8).digest(), "little"
    )


def _layout_hash(domain_ids: Sequence[int]) -> int:
    h = 0
    for domain_id in domain_ids:
        h = (h * 31 + domain_id % _HEADER_MODULUS) % _HEADER_MODULUS
    return h


class EvictionCoordinator:
    """Coordinate eviction decisions per process group, step and trigger domain.

    FREE_MEM TBEs on a rank share one trigger domain; every other TBE is its
    own. One SUM all_reduce per check step carries every due domain's votes and
    a header (due and registered domain counts and layout hashes, step, next
    check step) that each rank verifies, so a registry or cadence that differs
    across ranks raises instead of hanging or misapplying votes. A domain starts
    a round on every rank when some rank is urgent, or some rank wants one and
    every rank is idle (or the gate has been closed for
    `MAX_GATE_BLOCKED_CHECKS` checks), unless no rank's last round was
    sufficient within the backoff. A rank also wants a round when
    a member's TTL sweep interval has passed since the domain's last round on
    this rank, or none has run since the process started: wall-clock time only
    adds demand, it never gates.

    Invariants the collective depends on:
    - every rank enters it on the same steps, with an empty vote when nothing
      is due: the group's first check, then the step agreed at the previous one;
    - every decision input is a reduced tally or rank-invariant state;
    - `step` is rank-invariant. On a rank it may arrive out of order by up to
      `_MAX_STEP_LAG` across TBEs; see `_is_collective_step`.
    """

    def __init__(self) -> None:
        self._instances_by_pg: dict[
            dist.ProcessGroup | None,
            weakref.WeakKeyDictionary[EvictionParticipant, int],
        ] = {}
        self._last_seen_step_by_pg: dict[dist.ProcessGroup | None, int] = {}
        self._next_check_step_by_pg: dict[dist.ProcessGroup | None, int] = {}
        self._last_triggered_step: dict[tuple[dist.ProcessGroup | None, int], int] = {}
        self._blocked_checks: dict[tuple[dist.ProcessGroup | None, int], int] = {}
        self._urgent_backoff_warned: dict[tuple[dist.ProcessGroup | None, int], int] = (
            {}
        )
        self._last_round_time: dict[tuple[dist.ProcessGroup | None, int], float] = {}
        # Registration happens at the end of TBE construction, and construction
        # order is identical on every rank, so this sequence is a rank-invariant
        # ordering of members within a domain.
        self._next_seq: Iterator[int] = itertools.count()
        self._header_mismatch_pgs: set[dist.ProcessGroup | None] = set()

    def register(self, tbe: EvictionParticipant, pg: dist.ProcessGroup | None) -> None:
        if pg in self._last_seen_step_by_pg:
            # A TBE built after the group started checking belongs to a new
            # model, whose step count starts over. Construction happens at the
            # same point on every rank, so every rank resets together.
            self._reset(pg)
        instances = self._instances_by_pg.setdefault(pg, weakref.WeakKeyDictionary())
        instances[tbe] = next(self._next_seq)
        if len(self._domains(pg)) > MAX_DOMAINS:
            raise ValueError(
                f"More than {MAX_DOMAINS} KVZCH eviction trigger domains on one "
                "process group; raise MAX_DOMAINS"
            )

    def maybe_trigger(
        self,
        step: int,
        pg: dist.ProcessGroup | None,
        device: torch.device,
        free_mem_bytes: int | Callable[[], int],
    ) -> None:
        if step == 0 or not self._is_collective_step(pg, step):
            return

        # Which domains are due is a pure function of `step` plus per-table
        # config, so every rank derives the same list -- and the header check
        # below fails fast if one does not. The list may be empty.
        registered = self._domains(pg)
        domains = [
            domain
            for domain in registered
            if any(tbe.should_check_eviction(step) for tbe in domain.tbes)
        ]
        next_check = self._next_check_step(pg, step)

        needs_free_mem = any(
            tbe.uses_free_memory_trigger() or tbe.eviction_free_mem_floor_bytes > 0
            for domain in domains
            for tbe in domain.tbes
        )
        sampled_free_mem_bytes = (
            free_mem_bytes()
            if needs_free_mem and callable(free_mem_bytes)
            else free_mem_bytes if isinstance(free_mem_bytes, int) else 0
        )

        now = time.monotonic()
        local_votes = [0] * VOTE_SLOTS
        local_header = self._header(registered, domains, step, next_check)
        local_votes[:HEADER_SLOTS] = local_header
        local_decisions: list[tuple[bool, bool, int, bool]] = []
        for i, domain in enumerate(domains):
            wants, idle, reading = self._domain_votes(
                domain.tbes, step, sampled_free_mem_bytes
            )
            ttl_due = self._ttl_sweep_due((pg, domain.id), domain, now)
            wants = wants or ttl_due
            sufficient = any(
                tbe.ssd_db.last_eviction_sufficient() for tbe in domain.tbes
            )
            urgent = any(
                0 < tbe.eviction_free_mem_floor_bytes
                and sampled_free_mem_bytes < tbe.eviction_free_mem_floor_bytes
                for tbe in domain.tbes
            )
            base = HEADER_SLOTS + SLOTS_PER_DOMAIN * i
            local_votes[base : base + SLOTS_PER_DOMAIN] = [
                int(wants),
                int(idle),
                int(sufficient),
                int(urgent),
            ]
            local_decisions.append((wants, idle, reading, ttl_due))

        world_size = dist.get_world_size(pg) if dist.is_initialized() else 1
        votes = torch.tensor(
            local_votes,
            dtype=torch.int64,
            device=self._collective_device(pg, device),
        )
        if world_size > 1:
            dist.all_reduce(votes, op=dist.ReduceOp.SUM, group=pg)

        tallies = votes.tolist()
        if not self._header_agrees(tallies[:HEADER_SLOTS], local_header, world_size):
            self._on_header_mismatch(pg, tallies[:HEADER_SLOTS], local_header, step)
            return
        self._header_mismatch_pgs.discard(pg)
        self._next_check_step_by_pg[pg] = next_check
        for i, domain in enumerate(domains):
            base = HEADER_SLOTS + SLOTS_PER_DOMAIN * i
            tally = _Tally(*tallies[base : base + SLOTS_PER_DOMAIN])
            outcome = self._decide(pg, domain, step, world_size, tally)
            if outcome == "URGENT_BACKOFF":
                self._warn_urgent_backoff(pg, domain, step, world_size, tally)
            if outcome not in ("TRIGGERED", "URGENT", "GATE_TIMEOUT"):
                continue
            for tbe in domain.tbes:
                tbe.ssd_db.trigger_feature_evict(force=True)
            self._last_round_time[(pg, domain.id)] = now
            local_wants, local_idle, local_read, local_ttl_due = local_decisions[i]
            logger.info(
                f"[KVZCH Evict] step={step} domain={domain.label} "
                f"mode={domain.tbes[0].eviction_trigger_mode_id()} "
                f"tbes={len(domain.tbes)} local_read={local_read} "
                f"local_wants={int(local_wants)} local_gate={int(local_idle)} "
                f"votes_wants={tally.wants}/{world_size} "
                f"votes_gate={tally.idle}/{world_size} outcome={outcome} "
                f"votes_sufficient={tally.sufficient}/{world_size} "
                f"votes_urgent={tally.urgent}/{world_size} "
                f"local_ttl_due={int(local_ttl_due)}"
            )

    def _decide(
        self,
        pg: dist.ProcessGroup | None,
        domain: _Domain,
        step: int,
        world_size: int,
        tally: _Tally,
    ) -> str:
        key = (pg, domain.id)
        in_backoff = self._in_backoff(key, domain, step, tally)
        if tally.urgent > 0:
            outcome = "URGENT_BACKOFF" if in_backoff else "URGENT"
        elif tally.wants == 0:
            outcome = "NO_DEMAND"
        elif in_backoff:
            outcome = "BACKOFF"
        elif tally.idle == world_size:
            outcome = "TRIGGERED"
        else:
            blocked = self._blocked_checks.get(key, 0) + 1
            if blocked < MAX_GATE_BLOCKED_CHECKS:
                self._blocked_checks[key] = blocked
                return "GATE_BLOCKED"
            outcome = "GATE_TIMEOUT"
        self._blocked_checks.pop(key, None)
        if outcome in ("TRIGGERED", "URGENT", "GATE_TIMEOUT"):
            self._last_triggered_step[key] = step
        return outcome

    def _in_backoff(
        self,
        key: tuple[dist.ProcessGroup | None, int],
        domain: _Domain,
        step: int,
        tally: _Tally,
    ) -> bool:
        """Whether no rank's last round of the domain was sufficient and the
        backoff since it was triggered has not elapsed."""
        last = self._last_triggered_step.get(key)
        if last is None or tally.sufficient > 0:
            return False
        return step - last < self._backoff(domain)

    @staticmethod
    def _backoff(domain: _Domain) -> int:
        return max(tbe.eviction_insufficient_backoff_batch for tbe in domain.tbes)

    def _warn_urgent_backoff(
        self,
        pg: dist.ProcessGroup | None,
        domain: _Domain,
        step: int,
        world_size: int,
        tally: _Tally,
    ) -> None:
        # Once per backoff window.
        key = (pg, domain.id)
        last = self._last_triggered_step[key]
        if self._urgent_backoff_warned.get(key) == last:
            return
        self._urgent_backoff_warned[key] = last
        logger.warning(
            f"[KVZCH Evict] step={step} domain={domain.label}: free memory is "
            f"below the floor on {tally.urgent}/{world_size} ranks, but no rank "
            f"evicted more than 100 rows in the round at step {last}; the next "
            f"urgent round waits until step {last + self._backoff(domain)}"
        )

    def _is_collective_step(self, pg: dist.ProcessGroup | None, step: int) -> bool:
        last = self._last_seen_step_by_pg.get(pg)
        if last is not None and last - _MAX_STEP_LAG <= step <= last:
            return False
        if last is not None and step < last:
            # The step count restarted without a new registration. The step is
            # rank-invariant, so every rank starts over together.
            self._reset(pg)
        self._last_seen_step_by_pg[pg] = step
        next_check = self._next_check_step_by_pg.get(pg)
        return next_check is None or step >= next_check

    def _reset(self, pg: dist.ProcessGroup | None) -> None:
        self._last_seen_step_by_pg.pop(pg, None)
        self._next_check_step_by_pg.pop(pg, None)
        for state in (
            self._last_triggered_step,
            self._blocked_checks,
            self._urgent_backoff_warned,
            self._last_round_time,
        ):
            for key in [k for k in state if k[0] == pg]:
                del state[key]

    def _next_check_step(self, pg: dist.ProcessGroup | None, step: int) -> int:
        intervals = {
            tbe.eviction_check_interval_batch
            for tbe in self._instances_by_pg.get(pg, {})
        }
        return min(((step // i + 1) * i for i in intervals), default=step + 1)

    def _ttl_sweep_due(
        self,
        key: tuple[dist.ProcessGroup | None, int],
        domain: _Domain,
        now: float,
    ) -> bool:
        intervals = [
            tbe.eviction_ttl_sweep_interval_s
            for tbe in domain.tbes
            if tbe.eviction_ttl_sweep_interval_s is not None
        ]
        if not intervals:
            return False
        last = self._last_round_time.get(key)
        if last is not None:
            return now - last >= min(intervals)
        # The clock lives in memory. If a restore kept the checkpointed row
        # timestamps, rows may already be past their TTL, and a job restarted
        # more often than the interval would never sweep them, so the first
        # check is due. Otherwise every row is younger than this process, none
        # can expire before a full TTL, and the clock starts now.
        if any(tbe.ssd_db.restores_row_timestamps() for tbe in domain.tbes):
            return True
        self._last_round_time[key] = now
        return False

    @staticmethod
    def _header(
        registered: Sequence[_Domain],
        domains: Sequence[_Domain],
        step: int,
        next_check: int,
    ) -> list[int]:
        n = len(domains)
        layout = _layout_hash([domain.id for domain in domains])
        s = step % _HEADER_MODULUS
        t = next_check % _HEADER_MODULUS
        m = len(registered)
        registry = _layout_hash([domain.id for domain in registered])
        return [
            n,
            n * n,
            layout,
            layout * layout,
            s,
            s * s,
            t,
            t * t,
            m,
            m * m,
            registry,
            registry * registry,
        ]

    @staticmethod
    def _header_agrees(
        reduced: Sequence[int], local: Sequence[int], world_size: int
    ) -> bool:
        # For each value x, Σx == W·x and Σx² == W·x² hold on a rank only if x
        # has zero variance across ranks, so every rank reaches the same answer.
        return all(r == world_size * x for r, x in zip(reduced, local))

    def _on_header_mismatch(
        self,
        pg: dist.ProcessGroup | None,
        reduced: Sequence[int],
        local: Sequence[int],
        step: int,
    ) -> None:
        """
        Every rank sees the mismatch. The first time, every rank collects
        garbage and votes again at the next step: a TBE that is unreachable but
        not yet collected stays registered until its rank's garbage collector
        runs, which differs per rank. A mismatch that survives the collection
        is a real divergence and raises on every rank.
        """
        detail = (
            f"at step {step}: local [n, n^2, layout, layout^2, step, step^2, "
            f"next check step, its square, registered n, its square, registry "
            f"layout, its square] = {list(local)}, "
            f"reduced = {list(reduced)}"
        )
        if pg in self._header_mismatch_pgs:
            raise RuntimeError(
                f"KVZCH eviction vote layout differs across ranks {detail}. Every "
                "rank must register the same eviction-enabled TBEs, with the "
                "same check intervals."
            )
        self._header_mismatch_pgs.add(pg)
        logger.warning(
            f"[KVZCH Evict] vote layout differs across ranks {detail}; "
            "collecting garbage and voting again at the next step"
        )
        gc.collect()
        self._next_check_step_by_pg[pg] = step + 1

    def _domains(self, pg: dist.ProcessGroup | None) -> list[_Domain]:
        """
        Group the registered TBEs of `pg` into trigger domains, ordered by
        stable domain id. Two TBEs with the same mode and table names are told
        apart by their registration order, which is identical on every rank.
        """
        domains: dict[int, _Domain] = {}
        occurrences: dict[str, int] = {}
        for tbe, _ in sorted(
            self._instances_by_pg.get(pg, {}).items(), key=lambda kv: kv[1]
        ):
            if tbe.uses_free_memory_trigger():
                key = _FREE_MEM_DOMAIN
            else:
                member = tbe.eviction_domain_key()
                occurrence = occurrences.get(member, 0)
                occurrences[member] = occurrence + 1
                key = f"{member}#{occurrence}"
            domain_id = _stable_id(key)
            domain = domains.get(domain_id)
            if domain is None:
                label = (
                    _FREE_MEM_DOMAIN
                    if key == _FREE_MEM_DOMAIN
                    else f"tbe-{domain_id & 0xFFFFFFFF:08x}"
                )
                domains[domain_id] = _Domain(domain_id, label, [tbe])
            else:
                domain.tbes.append(tbe)
        return [domains[domain_id] for domain_id in sorted(domains)]

    @staticmethod
    def _domain_votes(
        tbes: Sequence[EvictionParticipant], step: int, free_mem_bytes: int
    ) -> tuple[bool, bool, int]:
        """Returns (wants, idle, reading), where reading is diagnostic."""
        wants = False
        idle = True
        reading = 0
        for tbe in tbes:
            tbe_wants, tbe_idle, tbe_read = tbe.ssd_db.should_trigger_evict(
                step, free_mem_bytes
            )
            # Every member of a domain is force-triggered together, so every
            # member's readiness counts -- including members that are off-cadence
            # on this step.
            idle = idle and tbe_idle
            if tbe_wants and tbe.should_check_eviction(step):
                wants = True
                # Report the reading that actually asked for the round.
                reading = tbe_read
            elif reading == 0:
                reading = tbe_read
        return wants, idle, reading

    @staticmethod
    def _collective_device(
        pg: dist.ProcessGroup | None, device: torch.device
    ) -> torch.device:
        if not dist.is_initialized() or dist.get_backend(pg) == dist.Backend.GLOO:
            return torch.device("cpu")
        return device
