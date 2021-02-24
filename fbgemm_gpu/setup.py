import os
import shutil

from codegen.embedding_backward_code_generator import emb_codegen
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cur_dir = os.path.dirname(os.path.realpath(__file__))
cub_include_path = os.getenv("CUB_DIR")
build_codegen_path = "build/codegen"
py_path = "python"

# Get the long description from the relevant file
with open(os.path.join(cur_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

OPTIMIZERS = [
    "adagrad",
    "adam",
    "lamb",
    "lars_sgd",
    "partial_rowwise_adam",
    "partial_rowwise_lamb",
    "rowwise_adagrad",
    "sgd",
]

cpp_cpu_output_files = (
    [
        "gen_embedding_backward_dense_split_cpu.cpp",
    ]
    + [
        "gen_embedding_backward_split_{}_cpu.cpp".format(optimizer)
        for optimizer in OPTIMIZERS
    ]
    + [
        "gen_embedding_backward_{}_split_cpu.cpp".format(optimizer)
        for optimizer in OPTIMIZERS
    ]
)

cpp_cuda_output_files = (
    [
        "gen_embedding_forward_split_weighted_codegen_cuda.cu",
        "gen_embedding_forward_split_unweighted_codegen_cuda.cu",
        "gen_embedding_forward_dense_weighted_codegen_cuda.cu",
        "gen_embedding_forward_dense_unweighted_codegen_cuda.cu",
        "gen_embedding_backward_split_indice_weights_codegen_cuda.cu",
        "gen_embedding_backward_dense_indice_weights_codegen_cuda.cu",
        "gen_embedding_backward_dense_split_unweighted_cuda.cu",
        "gen_embedding_backward_dense_split_weighted_cuda.cu",
    ]
    + [
        "gen_embedding_backward_{}_split_{}_cuda.cu".format(optimizer, weighted)
        for optimizer in OPTIMIZERS
        for weighted in [
            "weighted",
            "unweighted",
        ]
    ]
    + [
        "gen_embedding_backward_split_{}.cpp".format(optimizer)
        for optimizer in OPTIMIZERS
    ]
)

py_output_files = ["lookup_{}.py".format(optimizer) for optimizer in OPTIMIZERS]


def generate_jinja_files():
    abs_build_path = os.path.join(cur_dir, build_codegen_path)
    if not os.path.exists(abs_build_path):
        os.makedirs(abs_build_path)
    emb_codegen(install_dir=abs_build_path, is_fbcode=False)

    dst_python_path = os.path.join(cur_dir, py_path)
    if not os.path.exists(dst_python_path):
        os.makedirs(dst_python_path)
    for filename in py_output_files:
        shutil.copy2(os.path.join(abs_build_path, filename), dst_python_path)
    shutil.copy2(os.path.join(cur_dir, "codegen", "lookup_args.py"), dst_python_path)


class FBGEMM_GPU_BuildExtension(BuildExtension.with_options(no_python_abi_suffix=True)):
    def build_extension(self, ext):
        generate_jinja_files()
        super().build_extension(ext)


setup(
    name="fbgemm_gpu",
    install_requires=[
        "torch",
        "Jinja2",
        "click",
        "hypothesis",
    ],
    version="0.0.1",
    long_description=long_description,
    ext_modules=[
        CUDAExtension(
            name="fbgemm_gpu",
            sources=[
                os.path.join(cur_dir, build_codegen_path, "{}".format(f))
                for f in cpp_cuda_output_files + cpp_cpu_output_files
            ]
            + [
                os.path.join(cur_dir, "codegen/embedding_forward_split_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_backward_dense_host_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_backward_dense_host.cpp"),
                os.path.join(cur_dir, "src/split_embeddings_cache_cuda.cu"),
                os.path.join(cur_dir, "src/split_table_batched_embeddings.cpp"),
                os.path.join(cur_dir, "src/cumem_utils.cu"),
                os.path.join(cur_dir, "src/cumem_utils_host.cpp"),
                os.path.join(cur_dir, "src/quantize_wrappers.cu"),
                os.path.join(cur_dir, "src/quantize_ops_host.cpp"),
            ],
            include_dirs=[
                cur_dir,
                os.path.join(cur_dir, "include"),
                cub_include_path,
            ],
        )
    ],
    cmdclass={"build_ext": FBGEMM_GPU_BuildExtension},
)
