#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Check Python source code contains Meta copyright header
"""

from __future__ import annotations

import os
import sys

import click


def process_header(header, comment):
    lines = header.split("\n")
    new_lines = []
    for line in lines:
        if line is None or line == "":
            new_lines.append(comment)
        else:
            new_lines.append(comment + " " + line)
    return "\n".join(new_lines) + "\n"


HEADER = """Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""
HEADER_lines = HEADER.splitlines()[1:]
PY_HEADER = process_header(HEADER, "#")
CPP_HEADER = process_header(HEADER, "//")


def dfs(root_path: str) -> list[str]:
    """DFS source code tree to find python files missing header

    Parameters
    ----------
    root_path : str
        root source directory path

    Returns
    -------
    list[str]
        file list missing header
    """
    ret = []
    for root, _, files in os.walk(root_path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            if path.endswith(".py"):
                with open(path) as fi:
                    src = fi.read()
                    flag = True
                    for line in HEADER_lines:
                        if line not in src:
                            flag = False
                            break
                    if not flag:
                        ret.append(path)
    return ret


def fix_header(file_list: list[str]) -> None:
    """Adding Meta header to to source files

    Parameters
    ----------
    file_list : list[str]
        file list missing header
    """
    for path in file_list:
        src = ""
        with open(path) as fi:
            src = fi.read()
        with open(path, "w") as fo:
            fo.write(PY_HEADER)
            fo.write(src)


@click.command()
@click.option(
    "--path", help="Root directory of source to be checked", required=True, type=str
)
@click.option(
    "--fixit", default=False, help="Fix missing header", required=False, type=bool
)
def check_header(path, fixit):
    ret = dfs(path)
    if len(ret) == 0:
        sys.exit(0)
    print("Need to add Meta header to the following files.")
    print("----------------File List----------------")
    for line in ret:
        print(line)
    print("-----------------------------------------")
    if fixit:
        fix_header(ret)
    sys.exit(1)


if __name__ == "__main__":
    check_header()
