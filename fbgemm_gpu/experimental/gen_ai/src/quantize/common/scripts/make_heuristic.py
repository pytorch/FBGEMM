# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def build_heuristic(
    file_path: str, threshold: float
) -> Dict[Tuple[int, ...], List[Tuple[str, str]]]:
    """
    Builds a heuristic from a set of profiling runs on a kernel.
    The heuristic is currently built in a greedy approach:
    1. For all problem shapes consider all kernels performing within (1+threshold) of the fastest kernel.
    2. For the above kernels, count how often it appeared across all problem shapes.
    3. When assigning a kernel to a problem shape, prioritize kernels that appear more often to minimize the number of kernels used.

    Assumptions:

    The ordering of the problem shape dimension is assumed to be (outer_dim, inner_shapes) where outer_dim varies but inner_shapes is fixed in a set of problem shapes.
    This is how we decide how to map a set of problem shapes into a heuristic.
    E.g. For the following problem shapes (5, 5120, 1024), (6, 5120, 1024), (7, 2048, 1024), (8, 2048, 1024) we would build a heuristic along:

    Problem Shape (5120, 1024):
      5: ...
      6: ...

    Problem Shape (2048, 1024):
      7: ...
      8: ...
    """
    # Inner problem shapes
    inner_shapes: Set[Tuple[int, ...]] = set()
    # Problem Shape -> Best kernel time
    best_times_ms: Dict[Tuple[int, ...], float] = {}
    # Kernels count across all problem shapes
    kernel_count: Dict[str, int] = defaultdict(int)
    # Problem Shape -> Candidate kernels
    kernel_candidates: Dict[Tuple[int, ...], Set[str]] = defaultdict(set)
    # Problem Shape -> Assigned kernel
    kernel_assignment: Dict[Tuple[int, ...], str] = {}
    # Inner problem shape -> (Outer Dim, Kernel)
    heuristics: Dict[Tuple[int, ...], List[Tuple[str, str]]] = {}

    with open(file_path, "r") as file:
        parsed_rows = []

        # Parse CSV and find the best time for each problem shape.
        rows = file.readlines()
        for row in rows:
            problem_shape_str, kernel, time_ms_str = row.split(",")
            problem_shape = tuple(int(x) for x in problem_shape_str.split("_"))
            time_ms = float(time_ms_str)

            inner_shapes.add(problem_shape[1:])
            best_times_ms[problem_shape] = (
                time_ms
                if problem_shape not in best_times_ms
                else min(best_times_ms[problem_shape], time_ms)
            )

            parsed_rows.append((problem_shape, kernel, time_ms))

        # Filter kernels for each problem shape based on the permitted threshold
        for problem_shape, kernel, time_ms in parsed_rows:
            if time_ms < (best_times_ms[problem_shape] * (1 + threshold)):
                kernel_candidates[problem_shape].add(kernel)
                kernel_count[kernel] += 1

    # Prefer kernels that are used more often
    kernel_order = sorted(kernel_count.keys(), key=kernel_count.get, reverse=True)

    for kernel in kernel_order:
        for problem_shape, candidates in kernel_candidates.items():
            if problem_shape not in kernel_assignment and kernel in candidates:
                kernel_assignment[problem_shape] = kernel

    for inner_shape in inner_shapes:
        outer_dims_and_kernel = sorted(
            [
                (problem_shape[0], kernel)
                for problem_shape, kernel in kernel_assignment.items()
                if problem_shape[1:] == inner_shape
            ],
            key=lambda x: x[0],
        )

        heuristic: List[Tuple[str, str]] = []
        last_outer_dim, last_kernel = outer_dims_and_kernel[0]
        for outer_dim, assigned_kernel in outer_dims_and_kernel[1:]:
            if last_kernel != assigned_kernel:
                heuristic.append((str(last_outer_dim), last_kernel))
            last_outer_dim, last_kernel = outer_dim, assigned_kernel
        heuristic.append(("else", last_kernel))

        heuristics[inner_shape] = heuristic

    return heuristics


# A basic codegen to make the if statements, customize as needed for your kernel.
def print_heuristic_cpp(
    heuristics: Dict[Tuple[int, ...], List[Tuple[str, str]]],
    varnames_arg: str,
) -> None:
    varnames = dict(enumerate(varnames_arg.split(",")))

    for inner_shape, outer_dims_and_kernels in heuristics.items():
        condition = " && ".join(
            [f"{varnames[idx + 1]} == {val}" for idx, val in enumerate(inner_shape)]
        )
        print(f"if ({condition}) {{")
        for idx, (outer_dim, kernel) in enumerate(outer_dims_and_kernels):
            if idx == 0:
                print(f"  if ({varnames[0]} <= {outer_dim}) {{")
            elif idx == len(outer_dims_and_kernels) - 1:
                print("  } else {")
            else:
                print(f"  }} else if ({varnames[0]} <= {outer_dim}) {{")
            print(f"    return {kernel};")
        print("  }")
        print("}\n")


def main():
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
    parser.add_argument(
        "--cpp_varnames",
        type=str,
        help="Variable names to use for C++ heuristic generation, comma separated in same order of the problem shape. E.g. M,N,K",
    )

    args = parser.parse_args()

    heuristic = build_heuristic(args.file_path, args.threshold)
    if args.cpp:
        if args.cpp_varnames is None:
            print("If setting --cpp must also set --cpp_varnames.")
            exit(1)
        print_heuristic_cpp(heuristic, args.cpp_varnames)
    else:
        for inner_shape, outer_dims in heuristic.items():
            print(f"Problem Shape: {inner_shape}")
            for outer_dim, assigned_kernel in outer_dims:
                print(f"  {outer_dim}: {assigned_kernel}")


if __name__ == "__main__":
    main()
