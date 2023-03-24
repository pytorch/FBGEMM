load("@rules_cc//cc:defs.bzl", "cc_library")

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
    ],
    includes = [
        "asmjit/",
        "src/",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
)
