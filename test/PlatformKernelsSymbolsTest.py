# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import unittest
from collections.abc import Iterator

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

ALLOWED: set[str] = {
    "_ZTIN6fbgemm23PlatformSpecificKernelsE",
    "_ZTSN6fbgemm23PlatformSpecificKernelsE",
}


def defined_non_local_symbols(path: str) -> Iterator[str]:
    with open(path, "rb") as f:
        for section in ELFFile(f).iter_sections():
            if (
                not isinstance(section, SymbolTableSection)
                or section["sh_type"] != "SHT_SYMTAB"
            ):
                continue
            for symbol in section.iter_symbols():
                if (
                    symbol["st_info"]["bind"] != "STB_LOCAL"
                    and symbol["st_shndx"] != "SHN_UNDEF"
                ):
                    yield symbol.name


def in_isa_namespace(symbol: str, isa: str) -> bool:
    mangled = symbol[symbol.find("_Z") :]
    namespace = f"6fbgemm{len(isa)}{isa}"
    return re.match(rf"_Z(T[VISHW]|GV)?Z?N[rVKO]*{namespace}", mangled) is not None


def is_violation(symbol: str, isa: str) -> bool:
    if symbol in ALLOWED:
        return False
    if "_Z" not in symbol:
        return not symbol.startswith("__")
    return not in_isa_namespace(symbol, isa)


class PlatformKernelsSymbolsTest(unittest.TestCase):
    def test_only_isa_namespace_symbols(self) -> None:
        if not os.environ["FBGEMM_PLATFORM_KERNELS_OPTIMIZED"]:
            self.skipTest("unoptimized builds keep std helpers out of line")
        objects = [
            entry.split("=", 1)
            for entry in os.environ["FBGEMM_PLATFORM_KERNELS_OBJECTS"].split()
        ]
        self.assertTrue(objects)
        for isa, path in objects:
            with self.subTest(isa=isa, object=os.path.basename(path)):
                violations = sorted(
                    symbol
                    for symbol in defined_non_local_symbols(path)
                    if is_violation(symbol, isa)
                )
                self.assertEqual(violations, [])
