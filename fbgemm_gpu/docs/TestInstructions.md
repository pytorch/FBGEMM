## Running FBGEMM_GPU

The tests (in the `test/` directoy) and benchmarks (in the `bench/` directory)
provide good examples on how to use FBGEMM_GPU.

### FBGEMM_GPU Tests

To run the tests after building / installing the FBGEMM_GPU package:

```sh
# From the /fbgemm_gpu/ directory
cd test

python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning split_table_batched_embeddings_test.py
python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning quantize_ops_test.py
python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning sparse_ops_test.py
python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning split_embedding_inference_converter_test.py
```

#### Testing with the CUDA Variant

For the FBGEMM_GPU CUDA package, GPUs will be automatically detected and used
for testing.  To run the tests and benchmarks on a GPU-capable device in
CPU-only mode, `CUDA_VISIBLE_DEVICES=-1` must be set in the environment:

```sh
CUDA_VISIBLE_DEVICES=-1 python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning split_table_batched_embeddings_test.py
```

#### Testing with the ROCm Variant

For ROCm machines, testing against a ROCm GPU needs to be enabled with
`FBGEMM_TEST_WITH_ROCM=1` set in the environment:

```sh
# From the /fbgemm_gpu/ directory
cd test
FBGEMM_TEST_WITH_ROCM=1 python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning split_table_batched_embeddings_test.py
```

### FBGEMM_GPU Benchmarks

To run the benchmarks:

```sh
# From the /fbgemm_gpu/ directory
cd bench

python split_table_batched_embeddings_benchmark.py uvm
```
