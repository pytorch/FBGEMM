load("@rules_cc//cc:defs.bzl", "cc_library")

CXX_FLAGS = select({
    "@bazel_tools//src/conditions:windows": [
        "/std:c++17"
    ],
    "//conditions:default": [
        "-std=c++17"
    ],
})

cc_library(
    name = "asmjit",
    srcs = glob([
        "src/asmjit/core/*.cpp",
        "src/asmjit/x86/*.cpp",
        "src/asmjit/arm/*.cpp",
    ]),
    hdrs = glob([
        "src/asmjit/x86/*.h",
        "src/asmjit/core/*.h",
        "src/asmjit/*.h",
        "src/asmjit/arm/*.h",
    ]),
    copts = [
        "-DASMJIT_STATIC",
        "-fno-tree-vectorize",
        "-fmerge-all-constants",
        "-DTH_BLAS_MKL",
    ] + CXX_FLAGS,
    includes = [
        "asmjit/",
        "src/",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
)
