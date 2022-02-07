# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def get_fbgemm_base_srcs():
    return [
        "src/GenerateI8Depthwise.cc",
        "src/RefImplementations.cc",
        "src/Utils.cc",
    ]

def get_fbgemm_generic_srcs(with_base = False):
    return [
        "src/EmbeddingSpMDM.cc",
        "src/EmbeddingSpMDMNBit.cc",
        "src/ExecuteKernel.cc",
        "src/ExecuteKernelU8S8.cc",
        "src/Fbgemm.cc",
        "src/FbgemmBfloat16Convert.cc",
        "src/FbgemmConv.cc",
        "src/FbgemmFPCommon.cc",
        "src/FbgemmFP16.cc",
        "src/FbgemmFloat16Convert.cc",
        "src/FbgemmI64.cc",
        "src/FbgemmSparseDense.cc",
        "src/FbgemmI8Spmdm.cc",
        "src/GenerateKernelDirectConvU8S8S32ACC32.cc",
        "src/GenerateKernel.cc",
        "src/GenerateKernelU8S8S32ACC16.cc",
        "src/GenerateKernelU8S8S32ACC16Avx512.cc",  # Acc16 AVX512 JIT code gen
        "src/GenerateKernelU8S8S32ACC16Avx512VNNI.cc",
        "src/GenerateKernelU8S8S32ACC32.cc",
        "src/GenerateKernelU8S8S32ACC32Avx512VNNI.cc",
        "src/GroupwiseConv.cc",
        "src/GroupwiseConvAcc32Avx2.cc",
        "src/GroupwiseConvAcc32Avx512.cc",
        "src/PackAMatrix.cc",
        "src/PackAWithIm2Col.cc",
        "src/PackAWithQuantRowOffset.cc",
        "src/PackAWithRowOffset.cc",
        "src/PackBMatrix.cc",
        "src/PackMatrix.cc",
        "src/PackWeightMatrixForGConv.cc",
        "src/PackWeightsForConv.cc",
        "src/PackWeightsForDirectConv.cc",
        "src/QuantUtils.cc",
        "src/RowWiseSparseAdagradFused.cc",
        "src/SparseAdagrad.cc",
        "src/spmmUtils.cc",
        "src/TransposeUtils.cc",
    ] + (get_fbgemm_base_srcs() if with_base else [])

def get_fbgemm_public_headers():
    return [
        "include/fbgemm/ConvUtils.h",
        "include/fbgemm/Fbgemm.h",
        "include/fbgemm/FbgemmBuild.h",
        "include/fbgemm/FbgemmConvert.h",
        "include/fbgemm/FbgemmEmbedding.h",
        "include/fbgemm/FbgemmFP16.h",
        "include/fbgemm/FbgemmFPCommon.h",
        "include/fbgemm/FbgemmI64.h",
        "include/fbgemm/FbgemmI8DepthwiseAvx2.h",
        "include/fbgemm/FbgemmI8DirectconvAvx2.h",
        "include/fbgemm/FbgemmI8Spmdm.h",
        "include/fbgemm/FbgemmPackMatrixB.h",
        "include/fbgemm/FbgemmSparse.h",
        "include/fbgemm/OutputProcessing-inl.h",
        "include/fbgemm/PackingTraits-inl.h",
        "include/fbgemm/QuantUtils.h",
        "include/fbgemm/QuantUtilsAvx2.h",
        "include/fbgemm/QuantUtilsAvx512.h",
        "include/fbgemm/spmmUtils.h",
        "include/fbgemm/spmmUtilsAvx2.h",
        "include/fbgemm/Utils.h",
        "include/fbgemm/UtilsAvx2.h",
        "include/fbgemm/Types.h",
    ]

def get_fbgemm_avx2_srcs(msvc = False):
    return [
        #All the source files that either use avx2 instructions statically
        "src/EmbeddingSpMDMAvx2.cc",
        "src/FbgemmBfloat16ConvertAvx2.cc",
        "src/FbgemmFloat16ConvertAvx2.cc",
        "src/FbgemmI8Depthwise3DAvx2.cc",
        "src/FbgemmI8DepthwiseAvx2.cc",
        "src/FbgemmI8DepthwisePerChannelQuantAvx2.cc",
        "src/FbgemmSparseDenseAvx2.cc",
        "src/FbgemmSparseDenseInt8Avx2.cc",
        "src/OptimizedKernelsAvx2.cc",
        "src/PackDepthwiseConvMatrixAvx2.cc",
        "src/QuantUtilsAvx2.cc",
        "src/spmmUtilsAvx2.cc",
        "src/UtilsAvx2.cc",
    ]

def get_fbgemm_inline_avx2_srcs(msvc = False):
    return [
        #FP16 kernels contain inline assembly and inline assembly syntax for MSVC is different.
        "src/FbgemmFP16UKernelsAvx2.cc" if not msvc else "src/FbgemmFP16UKernelsIntrinsicAvx2.cc",
    ]

def get_fbgemm_avx512_srcs(msvc = False):
    return [
        #All the source files that use avx512 instructions statically
        "src/FbgemmBfloat16ConvertAvx512.cc",
        "src/EmbeddingSpMDMAvx512.cc",
        "src/FbgemmFloat16ConvertAvx512.cc",
        "src/FbgemmSparseDenseAvx512.cc",
        "src/FbgemmSparseDenseInt8Avx512.cc",
        "src/FbgemmSparseDenseVectorInt8Avx512.cc",
        "src/QuantUtilsAvx512.cc",
        "src/UtilsAvx512.cc",
    ]

def get_fbgemm_inline_avx512_srcs(msvc = False):
    return [
        #FP16 kernels contain inline assembly and inline assembly syntax for MSVC is different.
        "src/FbgemmFP16UKernelsAvx512.cc" if not msvc else "src/FbgemmFP16UKernelsIntrinsicAvx512.cc",
        "src/FbgemmFP16UKernelsAvx512_256.cc" if not msvc else "src/FbgemmFP16UKernelsIntrinsicAvx512_256.cc",
    ]

def get_fbgemm_tests(skip_tests = []):
    return native.glob(["test/*Test.cc"], exclude = skip_tests)
