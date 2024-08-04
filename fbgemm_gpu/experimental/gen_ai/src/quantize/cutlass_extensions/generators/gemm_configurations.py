# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fmt: off

"""
CUTLASS extension for generating Gemm kernels configurations.
"""
import argparse
import logging
import os
from itertools import product

# imports from third-party/cutlass-3
import cutlass.python.cutlass_library.library as library

# imports from cutlass_extensions
from .cutlass_extensions import CudaToolkitVersionSatisfies, FusionKind
from .gemm_operation import GemmOperation
from .manifest import Manifest

_LOGGER = logging.getLogger(__name__)


def numeric_log_level(log_level: str) -> int:
    """
    Converts the string identifier of the log level into the numeric identifier used
    in setting the log level

    :param x: string representation of log level (e.g., 'INFO', 'DEBUG')
    :type x: str

    :return: numeric representation of log level
    :rtype: int
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    return numeric_level
    
################################################################################################################################################

def CreateGemmUniversal3xOperator(
    manifest, layouts, tile_descriptions, data_types,
    fusion_kind=FusionKind.NoneScaling,
    schedules = None,
    epilogue_functor=library.EpilogueFunctor.LinearCombination,
    swizzling_functor=library.SwizzlingFunctor.Identity1,
    tile_schedulers=[library.TileSchedulerType.Persistent]):

    """ Generates 3.0 API based GemmUniversal API kernels. Alignment constraints are folded in with layouts. """

    if type(data_types) is dict:
        data_types = [data_types]

    for s in schedules:
        assert (len(s) == 2)

    operations = []

    combinations = product(layouts, tile_descriptions, data_types, schedules, tile_schedulers)
    for layout, tile_description, data_type, schedules, tile_scheduler in combinations:
        kernel_schedule, epilogue_schedule = schedules
        A = library.TensorDescription(data_type["a_type"], layout[0][0], layout[0][1], library.ComplexTransform.none)
        B = library.TensorDescription(data_type["b_type"], layout[1][0], layout[1][1], library.ComplexTransform.none)
        C = library.TensorDescription(data_type["c_type"], layout[2][0], layout[2][1])
        D = library.TensorDescription(data_type["d_type"], layout[2][0], layout[2][1])

        gemm_op_extra_args = {}
        gemm_kind = library.GemmKind.Universal3x
        element_compute = data_type.get("epi_type", data_type["acc_type"])

        operation = GemmOperation(
            gemm_kind, tile_description.minimum_compute_capability,
            tile_description, A, B, C, element_compute, fusion_kind, 
            epilogue_functor, swizzling_functor, D,
            kernel_schedule, epilogue_schedule, tile_scheduler, **gemm_op_extra_args)

        manifest.append(operation)
        operations.append(operation)

    return operations

################################################################################################################################################
#  CUTLASS Extensions GEMM Configurations
################################################################################################################################################

def GenerateF8F8BF16GemmWithTensorWiseScaling(manifest, cuda_version):
    """ Append (BF16 <= FP8 * FP8 + F32) GEMM with TensorWise Scaling to the manifest. """
    if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
        return

    # layouts for ABC and their alignments
    layouts = [
        [[library.LayoutType.RowMajor, 16], [library.LayoutType.ColumnMajor, 16], [library.LayoutType.ColumnMajor, 1],],  # TN Layout with N output.
        #[[library.LayoutType.RowMajor, 16], [library.LayoutType.ColumnMajor, 16], [library.LayoutType.RowMajor, 1],],  # TN Layout with T output.
    ]

    math_instructions = [
        # inst 64x128x32
        library.MathInstruction(
            [64, 128, 32],
            library.DataType.e4m3,
            library.DataType.e4m3,
            library.DataType.f32,
            library.OpcodeClass.TensorOp,
            library.MathOperation.multiply_add,
        ),
    ]

    min_cc = 90
    max_cc = 90

    for math_inst in math_instructions:
        # Datatype (BF16 <= FP8 * FP8 + F32)
        data_types = [
            {
                "a_type": math_inst.element_a,
                "b_type": math_inst.element_b,
                "c_type": library.DataType.bf16,
                "d_type": library.DataType.bf16,
                "acc_type": math_inst.element_accumulator,
                "epi_type": math_inst.element_accumulator,
            },
        ]

        # Threadblock shape and cluster shape
        tile_descriptions_small = [
            # 64x128x128
            library.TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
              0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
            library.TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
              0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
        ]
        tile_descriptions_large = [
            # 128x128x128
            library.TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
              0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
            library.TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
              0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
        ]
        for data_type in data_types:
            CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions_small, data_type, FusionKind.TensorWise,
                [[library.KernelScheduleType.TmaWarpSpecialized,            library.EpilogueScheduleType.NoSmemWarpSpecialized],
                [library.KernelScheduleType.TmaWarpSpecializedFP8FastAccum, library.EpilogueScheduleType.NoSmemWarpSpecialized]])

            CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions_large, data_type, FusionKind.TensorWise,
                [[library.KernelScheduleType.TmaWarpSpecialized,            library.EpilogueScheduleType.NoSmemWarpSpecialized],
                [library.KernelScheduleType.TmaWarpSpecializedFP8FastAccum, library.EpilogueScheduleType.NoSmemWarpSpecialized]])

################################################################################################################################################

# This function for defining the ArgumentParser is used to make it easy for the CUTLASS Python interface
# to leverage the functionality in this file without running this script via a shell prompt.
def define_parser():
    # get the current directory to use as the default build directory.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    default_build_dir = os.path.join(parent_dir, "kernel_library")

    # parser commandline arguments
    parser = argparse.ArgumentParser(description="Generates device kernel registration code for CUTLASS GEMM kernels")
    parser.add_argument("--operations", default="all", help="Specifies the operation to generate (gemm, all)")
    parser.add_argument("--build-dir", default=".", required=False, help="CUTLASS top-level build directory")
    parser.add_argument("--curr-build-dir", default=default_build_dir,
                        help="CUTLASS current build directory. cmake files will be emitted in this directory")
    parser.add_argument("--architectures", default='80;90', help="Target compute architectures")
    parser.add_argument("--kernels", default='', help='Comma delimited list to filter kernels by name.')
    parser.add_argument("--ignore-kernels", default='', help='Comma delimited list of kernels to exclude from build.')
    parser.add_argument("--filter-by-cc", default='True', type=str, 
                        help='If enabled, kernels whose compute capability range is not satisfied by the build '
                        'target are excluded.')
    parser.add_argument("--cuda-version", default="12.0.0", help="Semantic version string of CUDA Toolkit")
    parser.add_argument('--kernel-filter-file',   type=str, default=None, required=False, help='Full path of filter file')
    parser.add_argument("--log-level", default="info", type=numeric_log_level, required=False, 
                        help="Logging level to be used by the generator script")
    return parser


def invoke_main() -> None:
    parser = define_parser()
    args = parser.parse_args()

    # Set the logging level based on the user-provided `--log-level` command-line option
    logging.basicConfig(level=args.log_level)

    # Collect all the kernel configurations required to be generated in a single manifest object
    manifest = Manifest(args)
    GenerateF8F8BF16GemmWithTensorWiseScaling(manifest, args.cuda_version)
    manifest.emit()

###################################################################################################
