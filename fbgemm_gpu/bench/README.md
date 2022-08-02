### Benchmarks

## TorchRec FusedTableBatchedEmbeddingBags

[Torchrec](https://pytorch.org/torchrec/) uses fbgemm_gpu embedding and embedding bag implementations for Fused, Batched, Quantized versions of embedding and embeddingbag (in addition to other kernels).
They have run benchmarks on FusedEmbeddingBagCollection, which is implemented with fbgemm_gpu's [`SplitTableBatchedEmbeddingBagsCodegen`](https://github.com/pytorch/FBGEMM/blob/253b8842eeb2b33e65f7e2a7cfb79923b0e46bd7/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py#L171). They benchmark utilizing UVM and UVM-caching.
The [results](https://github.com/pytorch/torchrec/tree/main/benchmarks) show between 13x and 23x usecase in DLRM embedding sizes.
