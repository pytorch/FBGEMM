# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class ProblemShape:
    """Represents a problem shape with M, N, K dimensions."""

    M: int
    N: int
    K: int

    @classmethod
    def from_tuple(cls, shape_tuple: Tuple[int, int, int]) -> "ProblemShape":
        """Create ProblemShape from a tuple."""
        return cls(M=shape_tuple[0], N=shape_tuple[1], K=shape_tuple[2])

    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to tuple for backwards compatibility."""
        return (self.M, self.N, self.K)


@dataclass(frozen=True)
class KernelResult:
    """Represents a parsed kernel profiling result."""

    problem_shape: ProblemShape
    kernel: str
    time_ms: float


@dataclass
class KEntry:
    """Represents a K dimension entry with its assigned kernel."""

    K: int
    kernel: str


@dataclass
class NEntry:
    """Represents an N dimension entry with its K entries."""

    N: int
    k_entries: List[KEntry]


@dataclass
class MEntry:
    """Represents an M dimension entry with its N entries."""

    M: int
    n_entries: List[NEntry]


@dataclass
class Heuristic:
    """Represents the complete heuristic structure."""

    m_entries: List[MEntry]


def get_kernel_assignment(file_path: str, threshold: float) -> Dict[ProblemShape, str]:
    """
    Assign kernels to problem shape from a set of profiling runs on a kernel.
    The heuristic is currently built in a greedy approach:
    1. For all problem shapes consider all kernels performing within (1+threshold) of the fastest kernel.
    2. For the above kernels, count how often it appeared across all problem shapes.
    3. When assigning a kernel to a problem shape, prioritize kernels that appear more often to minimize the number of kernels used.
    """
    kernel_results: List[KernelResult] = []
    best_times_ms: Dict[ProblemShape, float] = {}
    kernel_count: Dict[str, int] = defaultdict(int)
    kernel_candidates: Dict[ProblemShape, Set[str]] = defaultdict(set)
    kernel_assignment: Dict[ProblemShape, str] = {}

    with open(file_path, "r") as file:
        # Parse CSV and find the best time for each problem shape
        for row in file:
            problem_shape_str, kernel, time_ms_str = row.strip().split(",")
            shape_tuple = tuple(int(x) for x in problem_shape_str.split("_"))
            problem_shape = ProblemShape.from_tuple(shape_tuple)
            time_ms = float(time_ms_str)

            result = KernelResult(
                problem_shape=problem_shape, kernel=kernel, time_ms=time_ms
            )
            kernel_results.append(result)

            best_times_ms[problem_shape] = (
                time_ms
                if problem_shape not in best_times_ms
                else min(best_times_ms[problem_shape], time_ms)
            )

        # Filter kernels for each problem shape based on the permitted threshold
        for result in kernel_results:
            if result.time_ms < (best_times_ms[result.problem_shape] * (1 + threshold)):
                kernel_candidates[result.problem_shape].add(result.kernel)
                kernel_count[result.kernel] += 1

    # Prefer kernels that are used more often across all problem shapes
    while len(kernel_assignment) < len(kernel_candidates):
        top_kernel = sorted(kernel_count.keys(), key=kernel_count.get, reverse=True)[0]
        for problem_shape, candidates in kernel_candidates.items():
            if problem_shape not in kernel_assignment and top_kernel in candidates:
                kernel_assignment[problem_shape] = top_kernel
                # Adjust kernel count to reflect this problem shape was just assigned
                for candidate in candidates:
                    kernel_count[candidate] -= 1

    return kernel_assignment


def get_heuristic(kernel_assignment: Dict[ProblemShape, str]) -> Heuristic:
    """Build hierarchical heuristic structure from kernel assignments."""
    M_vals = sorted({problem_shape.M for problem_shape in kernel_assignment.keys()})
    N_vals = sorted({problem_shape.N for problem_shape in kernel_assignment.keys()})
    K_vals = sorted({problem_shape.K for problem_shape in kernel_assignment.keys()})

    m_entries = []

    for M in M_vals:
        n_entries = []
        for N in N_vals:
            k_entries = []
            for K in K_vals:
                assignment = kernel_assignment[ProblemShape(M, N, K)]
                # Check if we can merge with the previous K entry
                if k_entries and k_entries[-1].kernel == assignment:
                    k_entries[-1].K = K
                else:
                    k_entries.append(KEntry(K=K, kernel=assignment))

            # If there's only one K entry, set K to 0 to indicate "all K values" so we can de-duplicate
            if len(k_entries) == 1:
                k_entries[0].K = 0

            # Check if we can merge with the previous N entry
            if n_entries and n_entries[-1].k_entries == k_entries:
                n_entries[-1].N = N
            else:
                n_entries.append(NEntry(N=N, k_entries=k_entries))

        # Check if we can merge with the previous M entry
        if m_entries and m_entries[-1].n_entries == n_entries:
            m_entries[-1].M = M
        else:
            m_entries.append(MEntry(M=M, n_entries=n_entries))

    return Heuristic(m_entries=m_entries)


def print_heuristic_cpp(heuristic: Heuristic) -> None:
    for m_idx, m_entry in enumerate(heuristic.m_entries):
        if m_idx == 0:
            print(f"if (M <= {m_entry.M}) {{")
        elif m_idx == len(heuristic.m_entries) - 1:
            print("} else {")
        else:
            print(f"}} else if (M <= {m_entry.M}) {{")

        for n_idx, n_entry in enumerate(m_entry.n_entries):
            if n_idx == 0:
                print(f"  if (N <= {n_entry.N}) {{")
            elif n_idx == len(m_entry.n_entries) - 1:
                print("  } else {")
            else:
                print(f"  }} else if (N <= {n_entry.N}) {{")

            if len(n_entry.k_entries) == 1:
                print(f"    return {n_entry.k_entries[0].kernel};")
            else:
                for k_idx, k_entry in enumerate(n_entry.k_entries):
                    if k_idx == 0:
                        print(f"    if (K <= {k_entry.K}) {{")
                    elif k_idx == len(n_entry.k_entries) - 1:
                        print("    } else {")
                    else:
                        print(f"    }} else if (K <= {k_entry.K}) {{")
                    print(f"      return {k_entry.kernel};")
                    if k_idx == len(n_entry.k_entries) - 1:
                        print("    }")

            if n_idx == len(m_entry.n_entries) - 1:
                print("  }")

        if m_idx == len(heuristic.m_entries) - 1:
            print("}")


def print_heuristic(heuristic: Heuristic) -> None:
    for m_entry in heuristic.m_entries:
        print(f"M: {m_entry.M}")
        for n_entry in m_entry.n_entries:
            print(f" N: {n_entry.N}")
            for k_idx, k_entry in enumerate(n_entry.k_entries):
                k_val = k_entry.K if k_idx != len(n_entry.k_entries) - 1 else "else"
                print(f"  K: {k_val} -> {k_entry.kernel}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate heuristics from FBGEMM kernel tuning cache detailed info."
    )
    parser.add_argument(
        "--file-path",
        type=str,
        required=True,
        help="Path to the input data file generated by the FBGEMM tuning cache.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Kernels performing within --threshold of the best fastest kernel will be considered.",
    )
    parser.add_argument(
        "--cpp", action="store_true", help="Generate C++ code for the heuristic."
    )

    args = parser.parse_args()

    kernel_assignment = get_kernel_assignment(args.file_path, args.threshold)
    heuristic = get_heuristic(kernel_assignment)

    if args.cpp:
        print_heuristic_cpp(heuristic)
    else:
        print_heuristic(heuristic)


if __name__ == "__main__":
    main()
