#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import random
import string
import tempfile
import unittest
from pathlib import Path
from typing import BinaryIO, Optional, Union

import fbgemm_gpu
import torch

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)


class FileStoreTest(unittest.TestCase):
    def _test_filestore_readwrite(
        self,
        # pyre-fixme[2]
        store,  # FileStore
        input: Union[BinaryIO, torch.Tensor, Path],
        path: Optional[str] = None,
    ) -> None:
        """
        Generic FileStore routines to test reading and writing data

        Args:
            store (FileStore): The FileStore to test
            input (torch.Tensor | BinaryIO | Path): The data to write to the FileStore
            path (str, optional): The path to write the data to. If not provided, a random path will be generated.
        """
        if path is None:
            path = "".join(random.choices(string.ascii_letters, k=15))

        assert not store.exists(path), f"{path} already exists"
        store.write(path, input)
        assert store.exists(path), f"{path} does not exist"

        if isinstance(input, torch.Tensor):
            assert torch.load(store.read(path)).equal(input), "tensors do not match"

        elif isinstance(input, io.BytesIO) or isinstance(input, BinaryIO):
            input.seek(0)
            assert store.read(path).read() == input.read(), "byte streams do not match"

        elif isinstance(input, Path):
            assert (
                store.read(path).read() == input.read_bytes()
            ), "file contents do not match"

        store.remove(path)
        assert not store.exists(path), f"{path} is not removed"

    def test_filestore_oss_bad_bucket(self) -> None:
        """
        Test that OSS FileStore raises ValueError when an invalid bucket is provided
        """
        from fbgemm_gpu.utils import FileStore

        self.assertRaises(
            ValueError, FileStore, "".join(random.choices(string.ascii_letters, k=15))
        )

    def test_filestore_oss_binaryio(self) -> None:
        """
        Test that OSS FileStore can read and write binary data
        """
        from fbgemm_gpu.utils import FileStore

        self._test_filestore_readwrite(
            FileStore("/tmp"),
            io.BytesIO("".join(random.choices(string.ascii_letters, k=128)).encode()),
        )

    def test_filestore_oss_tensor(self) -> None:
        """
        Test that OSS FileStore can read and write tensors
        """
        from fbgemm_gpu.utils import FileStore

        self._test_filestore_readwrite(
            FileStore("/tmp"),
            torch.rand((random.randint(100, 1000), random.randint(100, 1000))),
        )

    def test_filestore_oss_file(self) -> None:
        """
        Test that OSS FileStore can read and write files
        """
        from fbgemm_gpu.utils import FileStore

        input = torch.rand((random.randint(100, 1000), random.randint(100, 1000)))
        infile = tempfile.NamedTemporaryFile()
        torch.save(input, infile)

        self._test_filestore_readwrite(FileStore("/tmp"), Path(infile.name))

    @unittest.skipIf(open_source, "Test does not apply to OSS")
    def test_filestore_fb_bad_bucket(self) -> None:
        """
        Test that FB FileStore raises ValueError when an invalid bucket is provided
        """
        from fbgemm_gpu.fb.utils import FileStore

        self.assertRaises(
            ValueError, FileStore, "".join(random.choices(string.ascii_letters, k=15))
        )

    @unittest.skipIf(open_source, "Test does not apply to OSS")
    def test_filestore_fb_binaryio(self) -> None:
        """
        Test that FB FileStore can read and write binary data
        """
        from fbgemm_gpu.fb.utils import FileStore

        self._test_filestore_readwrite(
            FileStore("tlparse_reports"),
            io.BytesIO("".join(random.choices(string.ascii_letters, k=128)).encode()),
            f"tree/{''.join(random.choices(string.ascii_letters, k=15))}.unittest",
        )

    @unittest.skipIf(open_source, "Test does not apply to OSS")
    def test_filestore_fb_tensor(self) -> None:
        """
        Test that FB FileStore can read and write tensors
        """
        from fbgemm_gpu.fb.utils import FileStore

        self._test_filestore_readwrite(
            FileStore("tlparse_reports"),
            torch.rand((random.randint(100, 1000), random.randint(100, 1000))),
            f"tree/{''.join(random.choices(string.ascii_letters, k=15))}.unittest",
        )

    @unittest.skipIf(open_source, "Test does not apply to OSS")
    def test_filestore_fb_file(self) -> None:
        """
        Test that FB FileStore can read and write files
        """
        from fbgemm_gpu.fb.utils import FileStore

        input = torch.rand((random.randint(100, 1000), random.randint(100, 1000)))
        infile = tempfile.NamedTemporaryFile()
        torch.save(input, infile)

        self._test_filestore_readwrite(
            FileStore("tlparse_reports"),
            Path(infile.name),
            f"tree/{''.join(random.choices(string.ascii_letters, k=15))}.unittest",
        )
