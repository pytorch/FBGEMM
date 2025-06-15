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
from typing import Dict, List


def parse_filters() -> List[Dict[str, List[str]]]:
    """
    Parse filters from command line arguments into a list of dictionaries.

    Each --filter flag corresponds to one dictionary group.
    Supports syntax like:
      --filter key1:val1,val2;key2:val3 --filter key3:val4

    Returns:
      List[Dict[str, List[str]]]: A list of filter groups.
    """
    parser = argparse.ArgumentParser(
        description="Parse filters into a list of grouped dictionaries."
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Filter group(s) of the format key1:value_list_1[;key2:value_list_2], "
        "where value_list is of the format val1,[val2]. "
        "Multiple --filter flags are allowed.",
    )
    args = parser.parse_args()

    result = []

    for group in args.filter:
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


def and_match(coordinate: Dict[str, str], query: Dict[str, List[str]]) -> bool:
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
    coordinate: Dict[str, str], queries: List[Dict[str, List[str]]]
) -> bool:
    """
    Check if a build matrix coordinate matches any one of the queries.
    """
    return any([and_match(coordinate, query) for query in queries])


def main():
    # Parse the filter rules
    filter_rules = parse_filters()
    print(filter_rules, file=sys.stderr)

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
    new_matrix = {"include": new_matrix_entries}

    # Print the filtered matrix
    print(json.dumps(new_matrix))


if __name__ == "__main__":
    main()
