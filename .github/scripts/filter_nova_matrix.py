#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List


def parse_filter_rules() -> Dict[str, List[str]]:
    """
    Parse the filter rules from the command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Parse filters into a grouped dictionary."
    )
    parser.add_argument(
        "--filter",
        action="append",
        help="Filter in format key=value_list (e.g. name=val1,val2)",
    )
    args = parser.parse_args()

    filter_dict = defaultdict(list)

    if args.filter:
        for f in args.filter:
            if "=" not in f:
                raise ValueError(f"Invalid filter format: {f}. Expected key=value")

            key, values_str = f.split("=", 1)
            values = values_str.split(",")

            filter_dict[key].extend(values)

    return dict(filter_dict)


def passes_filter(
    coordinate: Dict[str, str], filter_dict: Dict[str, List[str]]
) -> bool:
    """
    Check if the build matrix coordinate passes the filter.
    """

    for key, values in filter_dict.items():
        if key not in coordinate:
            continue
        if coordinate[key] in values:
            return False

    return True


def main():
    # Parse the filter rules
    filter_rules = parse_filter_rules()

    # Exztract the full matrix
    full_matrix_string = os.environ["MAT"]
    full_matrix = json.loads(full_matrix_string)

    # Filter the matrix
    new_matrix_entries = [
        entry for entry in full_matrix["include"] if passes_filter(entry, filter_rules)
    ]
    new_matrix = {"include": new_matrix_entries}

    # Print the filtered matrix
    print(json.dumps(new_matrix))


if __name__ == "__main__":
    main()
