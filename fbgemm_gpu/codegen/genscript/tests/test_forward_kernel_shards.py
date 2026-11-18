#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.resources
import re
import runpy
import unittest
from importlib.resources.abc import Traversable

from deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.jinja_environment import (
    FORWARD_CACHE_TYPES,
    FORWARD_EMB_TYPES,
)

# The generator writes every shard into the codegen output directory, so a shard
# missing from a build manifest is never compiled: it fails only at load time.
_SHARDED_KERNEL = re.compile(
    r"^gen_embedding_forward_(?P<stem>\w+)_kernel_(?P<shard>[a-z0-9]+_[a-z0-9]+)\.cu$"
)

EXPECTED_SHARDS: set[str] = {
    f"{emb}_{cache}" for emb in FORWARD_EMB_TYPES for cache in FORWARD_CACHE_TYPES
}


def _files(root: Traversable) -> list[str]:
    names = []
    for entry in root.iterdir():
        if entry.is_dir():
            names.extend(_files(entry))
        else:
            names.append(entry.name)
    return names


def _shards_by_stem(filenames: list[str]) -> dict[str, set[str]]:
    shards: dict[str, set[str]] = {}
    for name in filenames:
        if match := _SHARDED_KERNEL.match(name):
            shards.setdefault(match["stem"], set()).add(match["shard"])
    return shards


def _buck_forward_files() -> list[str]:
    codegen = importlib.resources.files("deeplearning.fbgemm.fbgemm_gpu.codegen")
    return [
        name for name in _files(codegen) if name.startswith("gen_embedding_forward")
    ]


def _cmake_sources() -> dict[str, list[str]]:
    path = importlib.resources.files(__package__) / "tbe_sources.py"
    with importlib.resources.as_file(path) as tbe_sources:
        return runpy.run_path(str(tbe_sources))


class ForwardKernelShardsTest(unittest.TestCase):
    def test_buck_declares_every_shard(self) -> None:
        shards = _shards_by_stem(_buck_forward_files())
        self.assertTrue(shards, "no sharded forward kernels among Buck outputs")
        for stem, found in shards.items():
            self.assertEqual(found, EXPECTED_SHARDS, stem)

    def test_cmake_lists_every_shard(self) -> None:
        sources = _cmake_sources()
        self.assertEqual(set(sources["FORWARD_KERNEL_SHARDS"]), EXPECTED_SHARDS)
        shards = _shards_by_stem(sources["gen_gpu_files_forward_split"])
        self.assertTrue(shards, "no sharded forward kernels in tbe_sources.py")
        for stem, found in shards.items():
            self.assertEqual(found, EXPECTED_SHARDS, stem)

    def test_cmake_and_buck_list_the_same_forward_files(self) -> None:
        sources = _cmake_sources()
        cmake = set(sources["gen_cpu_files_forward_split"]) | set(
            sources["gen_gpu_files_forward_split"]
        )
        self.assertEqual(cmake, set(_buck_forward_files()))
