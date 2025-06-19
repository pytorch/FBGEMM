#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import io
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import torch

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileStore:
    """
    A basic file store implementation for easy data reads / writes / deletes.

    This class is intended to be used as a utility inside the FBGEMM_GPU codebase
    for consistent writing of tensors and other objects to the filesystem.

    Attribute:
        bucket (str): A directory in the filesystem.
    """

    bucket: str

    def __post_init__(self) -> None:
        if not os.path.isdir(self.bucket):
            raise ValueError(f"Directory {self.bucket} does not exist")

    def write(
        self,
        path: str,
        raw_input: Union[BinaryIO, torch.Tensor, Path],
        ttls: int = 864000,
    ) -> "FileStore":
        """
        Writes a binary stream, or a torch.Tensor to the file located at `path`
        (relative to `self.bucket`).

        Args:
            path (str): The path of the node or symlink to a directory.
            raw_input (BinaryIO | torch.Tensor | Path): The data to write.

            ttls (int): The time to live for the data in seconds. Defaults to
            10 days.

        Returns:
            self.  This allows for method-chaining.
        """

        filepath = f"{self.bucket}/{path}"
        event = f"writing to {filepath}"
        logger.info(f"FileStore: {event}")

        try:
            if os.path.isfile(filepath):
                raise FileExistsError(
                    f"File {filepath} already exists in the filesystem"
                )

            if isinstance(raw_input, torch.Tensor):
                torch.save(raw_input, filepath)

            elif isinstance(raw_input, Path):
                if not os.path.exists(raw_input):
                    raise FileNotFoundError(f"File {raw_input} does not exist")
                shutil.copyfile(raw_input, filepath)

            elif isinstance(raw_input, io.BytesIO) or isinstance(raw_input, BinaryIO):
                with open(filepath, "wb") as file:
                    raw_input.seek(0)
                    while chunk := raw_input.read(4096):  # Read 4 KB at a time
                        file.write(chunk)
            else:
                raise TypeError(f"Unsupported input type: {type(raw_input)}")

        except Exception as e:
            logger.error(f"FileStore: exception occurred when {event}: {e}")
            raise e

        return self

    def read(self, path: str) -> io.BytesIO:
        """
        Reads a file into a BytesIO object.

        Args:
            path (str): The path of the node or symlink to a directory (relative
            to `self.bucket`) to be read.

        Returns:
            Data from the file in BytesIO object format.
        """
        filepath = f"{self.bucket}/{path}"
        event = f"reading from {filepath}"
        logger.info(f"FileStore: {event}")

        try:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(
                    f"File {filepath} does not exist in the FileStore"
                )

            return io.BytesIO(open(filepath, "rb").read())

        except Exception as e:
            logger.error(f"FileStore: exception occurred when {event}: {e}")
            raise e

    def remove(self, path: str) -> "FileStore":
        """
        Removes a file or directory from the file store.

        Args:
            path (str): The path of the node or symlink to a directory (relative
            to `self.bucket`) to be removed.

        Returns:
            self.  This allows for method-chaining.
        """
        filepath = f"{self.bucket}/{path}"
        event = f"deleting {filepath}"
        logger.info(f"FileStore: {event}")

        try:
            if os.path.isfile(filepath):
                os.remove(filepath)

        except Exception as e:
            logger.error(f"Manifold: exception occurred when {event}: {e}")
            raise e

        return self

    def exists(self, path: str) -> bool:
        """
        Checks for existence of file in the file store.

        Args:
            path (str): The Manifold target path (relative to `self.bucket`).

        Returns:
            True if file exists, False otherwise.
        """
        filepath = f"{self.bucket}/{path}"
        return os.path.exists(filepath)

    def create_directory(self, path: str) -> "FileStore":
        """
        Creates a directory in the file store.

        Args:
            path (str): The path of the node or symlink to a directory (relative
            to `self.bucket`) to be created.

        Returns:
            self.  This allows for method-chaining.
        """
        filepath = f"{self.bucket}/{path}"
        event = f"creating directory {filepath}"
        logger.info(f"FileStore: {event}")

        try:
            if not os.path.exists(filepath):
                os.makedirs(filepath, exist_ok=True)
        except Exception as e:
            logger.error(f"FileStore: exception occurred when {event}: {e}")
            raise e

        return self

    def remove_directory(self, path: str) -> "FileStore":
        """
        Removes a directory from the file store.

        Args:
            path (str): The path of the node or symlink to a directory (relative
            to `self.bucket`) to be removed.

        Returns:
            self.  This allows for method-chaining.
        """
        filepath = f"{self.bucket}/{path}"
        event = f"deleting {filepath}"
        logger.info(f"FileStore: {event}")

        try:
            if os.path.isdir(filepath):
                os.rmdir(filepath)

        except Exception as e:
            logger.error(f"Manifold: exception occurred when {event}: {e}")
            raise e

        return self
