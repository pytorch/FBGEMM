# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import os
import sys
from dataclasses import dataclass
from typing import Dict, List

try:
    from .common import CodeTemplate
except:
    # pyre-ignore[21]
    from common import CodeTemplate


@dataclass
class TemplateParams:
    output_rows_per_thread: int
    input_rows_in_flight: int
    min_128b_rows: int
    max_128b_rows: int


@dataclass
class ElemType:
    cpp_type_name: str
    primitive_type: str
    bit_width: int
    template_params: List[TemplateParams]

    @property
    def enum_name(self) -> str:
        return f"{self.primitive_type}{self.bit_width}"


ELEM_TYPES = [
    ElemType(
        "float",
        "FP",
        32,
        [
            TemplateParams(2, 4, 0, 4),
            TemplateParams(2, 2, 4, 16),
            TemplateParams(1, 1, 16, 32),
            TemplateParams(1, 1, 32, 64),
        ],
    ),
    ElemType(
        "__half2",
        "FP",
        16,
        [
            TemplateParams(2, 8, 0, 2),
            TemplateParams(2, 8, 2, 4),
            TemplateParams(2, 4, 4, 8),
            TemplateParams(2, 2, 8, 16),
            TemplateParams(2, 1, 16, 32),
        ],
    ),
    ElemType(
        "uint32_t",
        "FP",
        8,
        [
            TemplateParams(2, 8, 0, 1),
            TemplateParams(2, 4, 1, 2),
            TemplateParams(2, 4, 2, 4),
            TemplateParams(2, 4, 4, 8),
            TemplateParams(2, 2, 8, 16),
        ],
    ),
    ElemType(
        "uint32_t",
        "INT",
        8,
        [
            TemplateParams(2, 8, 0, 1),
            TemplateParams(2, 4, 1, 2),
            TemplateParams(2, 4, 2, 4),
            TemplateParams(2, 4, 4, 8),
            TemplateParams(2, 2, 8, 16),
        ],
    ),
    ElemType(
        "uint32_t",
        "INT",
        4,
        [
            TemplateParams(4, 8, 0, 1),
            TemplateParams(2, 8, 1, 2),
            TemplateParams(1, 4, 2, 4),
            TemplateParams(1, 4, 4, 8),
        ],
    ),
    ElemType(
        "uint32_t",
        "INT",
        2,
        [
            TemplateParams(2, 16, 0, 1),
            TemplateParams(2, 8, 1, 2),
            TemplateParams(2, 8, 2, 4),
        ],
    ),
]

ELEM_TYPES_MAP: Dict[str, ElemType] = {etype.enum_name: etype for etype in ELEM_TYPES}


class ForwardQuantizedGenerator:
    @staticmethod
    def generate_nbit_kernel() -> None:
        # Generate the CUDA nbit (kernel) templates
        template = CodeTemplate.load(
            f"inference/embedding_forward_quantized_split_nbit_kernel_template.cu"
        )
        for weighted in [True, False]:
            for nobag in [True, False]:
                if not nobag or not weighted:
                    for emb_weight_type in ELEM_TYPES:
                        wdesc = f"{ 'weighted' if weighted else 'unweighted' }{ '_nobag' if nobag else '' }_{ emb_weight_type.enum_name.lower() }"
                        template.write(
                            f"gen_embedding_forward_quantized_split_nbit_kernel_{ wdesc }_codegen_cuda.cu",
                            weighted=weighted,
                            nobag=nobag,
                            emb_weight_type=emb_weight_type,
                        )

    @staticmethod
    def generate_nbit_host() -> None:
        # Generate the CUDA nbit (host) templates
        template = CodeTemplate.load(
            f"inference/embedding_forward_quantized_split_nbit_host_template.cu"
        )
        for weighted in [True, False]:
            for nobag in [True, False]:
                if not nobag or not weighted:
                    wdesc = f"{ 'weighted' if weighted else 'unweighted' }{ '_nobag' if nobag else '' }"
                    template.write(
                        f"gen_embedding_forward_quantized_split_nbit_host_{ wdesc }_codegen_cuda.cu",
                        weighted=weighted,
                        nobag=nobag,
                        type_map=ELEM_TYPES_MAP,
                    )

    @staticmethod
    def generate_nbit_cpu() -> None:
        # Generate the CPU templates
        template = CodeTemplate.load(
            f"inference/embedding_forward_quantized_cpu_template.cpp"
        )
        for weighted in [True, False]:
            template.write(
                f"gen_embedding_forward_quantized_{ 'weighted' if weighted else 'unweighted' }_codegen_cpu.cpp",
                weighted=weighted,
                type_map=ELEM_TYPES_MAP,
            )

    @staticmethod
    def generate() -> None:
        ForwardQuantizedGenerator.generate_nbit_kernel()
        ForwardQuantizedGenerator.generate_nbit_host()
        ForwardQuantizedGenerator.generate_nbit_cpu()


def main() -> None:
    ForwardQuantizedGenerator.generate()


if __name__ == "__main__":
    print(f"[GENERATE FORWARD QUANTIZED]: {sys.argv}")
    main()
