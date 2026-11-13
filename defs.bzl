# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def default_compiler_flags():
    return [
        "-pedantic",
        "-Wall",
        "-DFBGEMM_FBCODE",  # Used for enabling fbcode-specific code paths
    ] + select({
        "DEFAULT": [],
        "ovr_config//compiler:clang": [
            "-Werror",
            "-Wextra",
            "-Wno-gnu-zero-variadic-macro-arguments",
            "-Wno-c99-extensions",
            "-Wimplicit-fallthrough",
            "-Wignored-qualifiers",
            "-Wno-vla",
        ],
        "ovr_config//compiler:gcc": [
            "-Werror",
            "-Wextra",
            # Buck exposes Folly as non-system headers, where its intentional
            # `__int128` use triggers the `-pedantic` diagnostic.
            "-Wno-error=pedantic",
            "-fno-trapping-math",
        ]
        + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": [
                "-ftree-vectorize",
            ],
        }),
    })

def oss_ci_compiler_flags():
    # The CMake warning policy is in cmake/modules/CppLibrary.cmake.
    # Use mode/dev-nosan for cmake and mode/opt-clang for bazel; the build-mode
    # static/shared constraint controls the FBGEMM_STATIC definition.
    # Exercise OSS cache and autovec paths consistently in libraries and tests.
    return select({
        "DEFAULT": [],
        "fbcode//deeplearning/fbgemm/constraints:oss_ci_bazel": [
            "-UFBGEMM_FBCODE",
            "-UFBCODE_CAFFE2",
            "-UFBGEMM_MORE_SPECIALIZATION",
            "-std=c++20",
            "-O2",
            "-g0",
            "-DNDEBUG",
            "-fPIC",
            "-fstack-protector",
            "-U_FORTIFY_SOURCE",
            "-D_FORTIFY_SOURCE=1",
            "-fno-omit-frame-pointer",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wall",
            "-Wunused-but-set-parameter",
            "-Wno-free-nonheap-object",
        ] + select({
            "DEFAULT": [],
            "ovr_config//compiler:clang": ["-Wthread-safety", "-Wself-assign"],
        }),
        "fbcode//deeplearning/fbgemm/constraints:oss_ci_cmake": [
            "-UFBGEMM_FBCODE",
            "-UFBCODE_CAFFE2",
            "-UFBGEMM_MORE_SPECIALIZATION",
            "-std=c++20",
            "-O3",
            "-g0",
            "-DNDEBUG",
            "-fPIC",
            "-fno-omit-frame-pointer",
            "-fsanitize=address",
            "-DASMJIT_STATIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Waddress",
            "-Wenum-compare",
            "-Wmisleading-indentation",
            "-Wparentheses",
            "-Wpessimizing-move",
            "-Wunused-label",
            "-Wunused-local-typedefs",
            "-Wunused-value",
            "-Wimplicit-fallthrough",
            "-Wuninitialized",
            "-Wvexing-parse",
            "-Wmissing-braces",
            "-Wmismatched-tags",
            "-Waddress-of-packed-member",
            "-Wshadow",
            "-Wzero-as-null-pointer-constant",
            "-Wunused-variable",
            "-Wunused-const-variable",
            "-Wunused-but-set-variable",
            "-Wunused-function",
            "-Wunused-result",
            "-Wignored-qualifiers",
            "-Wtautological-compare",
            "-Wno-deprecated-declarations",
            "-Wno-deprecated-enum-enum-conversion",
            "-Wno-strict-aliasing",
            "-Wno-sign-compare",
            "-Wno-vla",
        ]
        + select({
            "DEFAULT": [],
            "ovr_config//compiler:clang": [
                "-Winline-asm",
                "-Wasm-operand-widths",
                "-Wmove",
                "-Winfinite-recursion",
                "-Wself-assign",
                "-Wnull-conversion",
                "-Wstring-concatenation",
                "-Wdeprecated-dynamic-exception-spec",
                "-Wdeprecated-this-capture",
                "-Wdeprecated-copy-with-user-provided-copy",
                "-Wundefined-var-template",
                "-Wgcc-compat",
                "-Wextra-qualification",
                "-Wstring-conversion",
                "-Wimplicitly-unsigned-literal",
                "-Wuninitialized-const-reference",
                "-Wthread-safety",
                "-Wdangling",
                "-Wshift-sign-overflow",
                "-Wunused-exception-parameter",
                "-Wheader-hygiene",
                "-Wshorten-64-to-32",
                "-Wdeprecated-implementations",
                "-Wsemicolon-before-method-body",
                "-Wimport-preprocessor-directive-pedantic",
                "-Wincompatible-function-pointer-types",
                "-Wambiguous-reversed-operator",
                "-Wbitwise-instead-of-logical",
                "-Wunreachable-code-fallthrough",
                "-Wunused-local-typedef",
                "-Wno-unused-command-line-argument",
                "-Wno-c99-extensions",
                "-Wno-gnu-zero-variadic-macro-arguments",
                "-Wno-error=deprecated-dynamic-exception-spec",
                "-Wno-error=shorten-64-to-32",
            ],
            "ovr_config//compiler:gcc": [
                "-Wno-error=array-bounds",
            ]
            + select({
                "DEFAULT": [],
                # Match the portable ARM build; GCC 11 lacks the SVE bridge.
                "ovr_config//cpu:arm64": ["-march=armv8-a", "-mtune=generic"],
            }),
        })
        + select({
            "DEFAULT": [],
            "ovr_config//build_mode/constraints:static": ["-DFBGEMM_STATIC"],
        }),
    })

def oss_ci_runtime_deps():
    # Shared libraries use weak stubs; the executable owns the ASAN runtime.
    return select({
        "DEFAULT": [],
        "fbcode//deeplearning/fbgemm/constraints:oss_ci_cmake": ["fbcode//deeplearning/fbgemm:oss_ci_asan_stubs"],
    })

def get_fbgemm_base_srcs():
    return [
        "src/GenerateI8Depthwise.cc",
        "src/JitPerfMap.cc",
        "src/RefImplementations.cc",
        "src/Utils.cc",
    ]

def get_fbgemm_generic_srcs(with_base = False):
    sources = [
        "src/EmbeddingSpMDM.cc",
        "src/EmbeddingSpMDMNBit.cc",
        "src/ExecuteKernel.cc",
        "src/ExecuteKernelU8S8.cc",
        "src/Fbgemm.cc",
        "src/FbgemmBfloat16Convert.cc",
        "src/FbgemmConv.cc",
        "src/FbgemmFPCommon.cc",
        "src/FbgemmFP16.cc",
        "src/fp32/FbgemmFP32.cc",
        "src/FbgemmFloat16Convert.cc",
        "src/FbgemmI64.cc",
        "src/FbgemmSparseDense.cc",
        "src/FbgemmI8Spmdm.cc",
        "src/FbgemmPackMatrixB.cc",
        "src/GenerateKernelDirectConvU8S8S32ACC32.cc",
        "src/GenerateKernel.cc",
        "src/GenerateKernelFP16FP32.cc",
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
        "src/PlatformSpecificKernels.cc",
        "src/QuantUtils.cc",
        "src/RowWiseSparseAdagradFused.cc",
        "src/SparseAdagrad.cc",
        "src/spmmUtils.cc",
        "src/TransposeUtils.cc",
    ] + (get_fbgemm_base_srcs() if with_base else [])

    return sources

def get_fbgemm_public_headers():
    return [
        "include/fbgemm/Assert.h",
        "include/fbgemm/ConvUtils.h",
        "include/fbgemm/Fbgemm.h",
        "include/fbgemm/FbgemmBuild.h",
        "include/fbgemm/FbgemmConvert.h",
        "include/fbgemm/FbgemmEmbedding.h",
        "include/fbgemm/detail/EmbeddingSpMDMFP8.h",
        "include/fbgemm/FbgemmFP16.h",
        "include/fbgemm/FbgemmFP32.h",
        "include/fbgemm/FbgemmFPCommon.h",
        "include/fbgemm/FbgemmI64.h",
        "include/fbgemm/FbgemmI8Depthwise.h",
        "include/fbgemm/FbgemmI8DirectconvAvx2.h",
        "include/fbgemm/FbgemmI8Spmdm.h",
        "include/fbgemm/FbgemmPackMatrixB.h",
        "include/fbgemm/FbgemmSparse.h",
        "include/fbgemm/FloatConversion.h",
        "include/fbgemm/OutputProcessing-inl.h",
        "include/fbgemm/PackingTraits-inl.h",
        "include/fbgemm/QuantUtils.h",
        "include/fbgemm/QuantUtilsAvx2.h",
        "include/fbgemm/QuantUtilsAvx512.h",
        "include/fbgemm/QuantUtilsNeon.h",
        "include/fbgemm/spmmUtils.h",
        "include/fbgemm/spmmUtilsAvx2.h",
        "include/fbgemm/SimdUtils.h",
        "include/fbgemm/Utils.h",
        "include/fbgemm/UtilsAvx2.h",
        "include/fbgemm/Types.h",
    ]

def get_fbgemm_platform_kernels_srcs():
    return [
        "src/isa_specific/FbgemmBfloat16ConvertKernels.cc",
        "src/isa_specific/PlatformSpecificKernelsIsa.cc",
    ]

def get_fbgemm_platform_kernels_isas(arch):
    return ["ref"] + {
        "aarch64": ["neon", "sve128"],
        "x86_64": ["avx2", "avx512"],
    }.get(arch, [])

def get_fbgemm_avx2_flags():
    return [
        "-m64",
        "-mavx2",
        "-mfma",
        "-mf16c",
    ]

def get_fbgemm_avx512_flags():
    return [
        "-m64",
        "-mavx2",
        "-mfma",
        "-mavx512f",
        "-mavx512bw",
        "-mavx512dq",
        "-mavx512vl",
        "-mavx512bf16",
    ]

def get_fbgemm_platform_kernels_flags(isa):
    return {
        "avx2": get_fbgemm_avx2_flags(),
        "avx512": get_fbgemm_avx512_flags(),
        "sve128": [
            "-march=armv8.5-a+sve2",
            "-msve-vector-bits=128",
        ],
    }.get(isa, [])

def get_fbgemm_avx2_srcs(buck = False):
    # These 3 depthwise files are also compiled in fbgemm_sve.  On ARM,
    # exclude them from fbgemm_avx2 to avoid duplicate symbols when
    # downstream targets pull in both fbgemm_avx2 and fbgemm_sve (the
    # latter is selected by the main fbgemm target on arm64).
    depthwise_srcs = [
        "src/FbgemmI8Depthwise3D.cc",
        "src/FbgemmI8Depthwise.cc",
        "src/PackDepthwiseConvMatrix.cc",
    ]

    common_srcs = [
        # All the source files that either use avx2 instructions statically
        "src/EmbeddingSpMDMAvx2.cc",
        "src/FbgemmBfloat16ConvertAvx2.cc",
        "src/FbgemmFloat16ConvertAvx2.cc",
        "src/FbgemmI8DepthwisePerChannelQuantAvx2.cc",
        "src/FbgemmSparseDenseAvx2.cc",
        "src/FbgemmSparseDenseInt8Avx2.cc",
        "src/OptimizedKernelsAvx2.cc",
        "src/QuantUtilsAvx2.cc",
        "src/spmmUtilsAvx2.cc",
        "src/UtilsAvx2.cc",
    ]

    if buck:
        return common_srcs + select({
            "DEFAULT": depthwise_srcs,
            "ovr_config//cpu:arm64": [],  # Provided by fbgemm_sve on ARM
        })
    return common_srcs + depthwise_srcs

def get_fbgemm_inline_avx2_srcs(msvc = False, buck = False):
    intrinsics_srcs = ["src/FbgemmFP16UKernelsIntrinsicAvx2.cc"]

    # FP16 kernels contain inline assembly and inline assembly syntax for MSVC is different.
    asm_srcs = [
        "src/fp32/FbgemmFP32UKernelsAvx2.cc",
        "src/FbgemmFP16UKernelsAvx2.cc",
    ]
    if buck:
        return select({
            "DEFAULT": asm_srcs,
            "ovr_config//compiler:cl": intrinsics_srcs,
            "ovr_config//cpu:arm64": intrinsics_srcs,
        })
    return asm_srcs if not msvc else intrinsics_srcs

def get_fbgemm_avx512_srcs():
    return [
        # All the source files that use avx512 instructions statically
        "src/FbgemmBfloat16ConvertAvx512.cc",
        "src/EmbeddingSpMDMAvx512.cc",
        "src/FbgemmFloat16ConvertAvx512.cc",
        "src/FbgemmSparseDenseAvx512.cc",
        "src/FbgemmSparseDenseInt8Avx512.cc",
        "src/FbgemmSparseDenseVectorInt8Avx512.cc",
        "src/QuantUtilsAvx512.cc",
        "src/UtilsAvx512.cc",
    ]

def get_fbgemm_inline_avx512_srcs(msvc = False, buck = False):
    intrinsics_srcs = [
        "src/FbgemmFP16UKernelsIntrinsicAvx512.cc",
        "src/FbgemmFP16UKernelsIntrinsicAvx512_256.cc",
    ]
    asm_srcs = [
        "src/FbgemmFP16UKernelsAvx512.cc",
        "src/FbgemmFP16UKernelsAvx512_256.cc",
        "src/fp32/FbgemmFP32UKernelsAvx512.cc",
        "src/fp32/FbgemmFP32UKernelsAvx512_256.cc",
    ]
    if buck:
        return select({
            "DEFAULT": asm_srcs,
            "ovr_config//compiler:cl": intrinsics_srcs,
            "ovr_config//cpu:arm64": intrinsics_srcs,
        })
    return asm_srcs if not msvc else intrinsics_srcs

def get_fbgemm_inline_sve_srcs(msvc = False, buck = False):
    srcs = [
        "src/FbgemmFP16UKernelsSve128.cc",
        "src/UtilsSve.cc",
        "src/FbgemmFloat16ConvertSVE.cc",
        "src/PackDepthwiseConvMatrix.cc",
        "src/FbgemmI8Depthwise3D.cc",
        "src/FbgemmI8Depthwise.cc",
    ]

    if buck:
        # FP16 kernels contain inline assembly and inline assembly syntax for MSVC is different.
        srcs += select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": [
                "src/FbgemmFloat16ConvertSVE.cc",
            ],
        })
        return select({
            "DEFAULT": srcs,
            "ovr_config//compiler:cl": srcs,
            "ovr_config//cpu:arm64": srcs,
        })

    else:
        return srcs

def get_fbgemm_inline_neon_srcs(msvc = False, buck = False):
    intrinsics_srcs = [
        "src/UtilsNeon.cc",
        "src/FbgemmFloat16ConvertNEON.cc",
    ]

    # FP16 kernels contain inline assembly and inline assembly syntax for MSVC is different.
    asm_srcs = [
        "src/UtilsNeon.cc",
        "src/FbgemmFloat16ConvertNEON.cc",
        "src/KleidiAIFP16UKernelsNeon.cc",
        "src/fp32/KleidiAIFP32UKernelsNeon.cc",
        "src/QuantUtilsNeon.cc",
    ]
    if buck:
        return select({
            "DEFAULT": asm_srcs,
            "ovr_config//compiler:cl": intrinsics_srcs,
            "ovr_config//cpu:arm64": asm_srcs,
        })
    return asm_srcs if not msvc else intrinsics_srcs

def get_fbgemm_autovec_srcs():
    return [
        "src/EmbeddingSpMDMAutovec.cc",
        "src/EmbeddingStatsTracker.cc",
    ]

def get_fbgemm_tests(skip_tests = []):
    return native.glob(["test/*Test.cc"], exclude = skip_tests)

def select_fbgemm_codegen_mode(inference, training):
    return select({
        "//deeplearning/fbgemm/constraints:codegen_mode[inference]": inference,
        "//deeplearning/fbgemm/constraints:codegen_mode[training]": training,
        "DEFAULT": select({
            "//deeplearning/fbgemm/constraints:fbgemm_codegen_inference_mode_enabled": inference,
            "//deeplearning/fbgemm/constraints:fbgemm_codegen_inference_mode_enabled_capitalized": inference,
            "//deeplearning/fbgemm/constraints:fbgemm_codegen_inference_mode_enabled_numeric": inference,
            "DEFAULT": training,
        }),
    })
