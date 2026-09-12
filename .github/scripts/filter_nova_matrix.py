#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and adjust a Nova build matrix."
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Filter group(s) of the format key1:value_list_1[;key2:value_list_2], "
        "where value_list is of the format val1,[val2]. "
        "Multiple --filter flags are allowed.",
    )
    parser.add_argument(
        "--runner",
        action="append",
        default=[],
        help="Runner override(s) of the format <selector>=<runner_label>, where "
        "<selector> uses the same grammar as --filter. Coordinates matching the "
        "selector have their validation_runner replaced with <runner_label>. "
        "Multiple --runner flags are allowed; the first match wins.",
    )
    return parser.parse_args()


def parse_filters(groups: list[str]) -> list[dict[str, list[str]]]:
    """
    Parse filter groups into a list of dictionaries.

    Each group corresponds to one dictionary.
    Supports syntax like:
      key1:val1,val2;key2:val3

    Returns:
      list[dict[str, list[str]]]: A list of filter groups.
    """
    result = []

    for group in groups:
        filter_dict = defaultdict(list)
        parts = group.split(";")
        for part in parts:
            if ":" not in part:
                raise ValueError(
                    f"Invalid filter format: {part}. Expected key:value(s)"
                )

            key, values_str = part.split(":", 1)
            values = values_str.split(",")
            filter_dict[key].extend(values)

        result.append(dict(filter_dict))

    return result


def and_match(coordinate: dict[str, str], query: dict[str, list[str]]) -> bool:
    """
    Check if a build matrix coordinate matches all the query parameters.
    """
    for key, values in query.items():
        if key not in coordinate:
            continue
        if coordinate[key] not in values:
            return False
    return True


def query_match(
    coordinate: dict[str, str], queries: list[dict[str, list[str]]]
) -> bool:
    """
    Check if a build matrix coordinate matches any one of the queries.
    """
    return any([and_match(coordinate, query) for query in queries])


def parse_runner_overrides(specs: list[str]) -> list[tuple[dict[str, list[str]], str]]:
    """
    Parse runner overrides of the format <selector>=<runner_label> into
    (selector, runner_label) pairs, where <selector> uses the --filter grammar.

    Returns:
      list[tuple[dict[str, list[str]], str]]: A list of (selector, runner) pairs.
    """
    result = []

    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid runner override format: {spec}. "
                "Expected <selector>=<runner_label>"
            )

        selector_str, runner = spec.split("=", 1)
        [selector] = parse_filters([selector_str])
        result.append((selector, runner))

    return result


def strict_and_match(coordinate: dict[str, str], query: dict[str, list[str]]) -> bool:
    """
    Check if a build matrix coordinate matches all the query parameters, and
    carries every key the query selects on.

    This is deliberately stricter than `and_match`.  Filtering is subtractive,
    so treating an absent key as satisfied there only ever drops coordinates.
    A runner override is targeted, and the same leniency would silently
    retarget any coordinate that never declared the field being selected on.
    """
    return all(coordinate.get(key) in values for key, values in query.items())


def apply_runner_overrides(
    coordinates: list[dict[str, str]],
    overrides: list[tuple[dict[str, list[str]], str]],
) -> None:
    """
    Rewrite the validation_runner of every coordinate matching an override
    selector, in place.  The first matching override wins.
    """
    for coordinate in coordinates:
        for selector, runner in overrides:
            if strict_and_match(coordinate, selector):
                coordinate["validation_runner"] = runner
                break


def main():
    args = parse_args()

    # Parse the filter rules and the runner overrides
    filter_rules = parse_filters(args.filter)
    runner_overrides = parse_runner_overrides(args.runner)
    print(filter_rules, runner_overrides, file=sys.stderr)

    # Exztract the full matrix
    full_matrix_string = os.environ["MAT"]
    full_matrix = json.loads(full_matrix_string)

    # Filter the matrix
    new_matrix_entries = [
        coordinate
        for coordinate in full_matrix["include"]
        # Filter out build matrix coordinates if they match one of the queries
        if not query_match(coordinate, filter_rules)
    ]

    # Move the surviving coordinates onto bigger machines where requested
    apply_runner_overrides(new_matrix_entries, runner_overrides)

    new_matrix = {"include": new_matrix_entries}

    # Print the filtered matrix
    print(json.dumps(new_matrix))


if __name__ == "__main__":
    main()
