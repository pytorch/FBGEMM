# FBGEMM

[![FBGEMM CI](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_ci.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_ci.yml)

FBGEMM (Facebook GEneral Matrix Multiplication) is a low-precision,
high-performance matrix-matrix multiplications and convolution library for
server-side inference.

The library provides efficient low-precision general matrix multiplication for
small batch sizes and support for accuracy-loss minimizing techniques such as
row-wise quantization and outlier-aware quantization. FBGEMM also exploits
fusion opportunities in order to overcome the unique challenges of matrix
multiplication at lower precision with bandwidth-bound operations.

FBGEMM is used as a backend of Caffe2 and PyTorch quantized operators for x86 machines:

  * Caffe2: https://github.com/pytorch/pytorch/tree/master/caffe2/quantization/server
  * PyTorch: https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/quantized/cpu

See the full [Documentation](https://pytorch.org/FBGEMM) for more information
on building, installing, and developing with FBGEMM, as well as the most
up-to-date support matrix and API documentation for this library.

### What's New?

* [New Features and Recent Improvements](https://github.com/pytorch/FBGEMM/wiki/Recent-feature-additions-and-improvements-in-FBGEMM) (January, 2020)

### Citation

For a high-level overview, design philosophy and brief descriptions of various
parts of FBGEMM please see [our blog post](https://code.fb.com/ml-applications/fbgemm).

For those looking for the appropriate article to cite regarding FBGEMM, we
recommend citing our [paper](https://arxiv.org/pdf/2101.05615.pdf):

```
@article{fbgemm,
  title={FBGEMM: Enabling High-Performance Low-Precision Deep Learning Inference},
  author={Khudia, Daya and Huang, Jianyu and Basu, Protonu and Deng, Summer and Liu, Haixin and Park, Jongsoo and Smelyanskiy, Mikhail},
  journal={arXiv preprint arXiv:2101.05615},
  year={2021}
}
```


## Join the FBGEMM community

For questions, support, news updates, or feature requests, please feel free to:

* File a ticket in [GitHub Issues](https://github.com/pytorch/FBGEMM/issues)
* Post a discussion in [GitHub Discussions](https://github.com/pytorch/FBGEMM/discussions)
* Reach out to us on the `#fbgemm` channel in [PyTorch Slack](https://bit.ly/ptslack)

For contributions, please see the [`CONTRIBUTING`](../CONTRIBUTING.md) file for
ways to help out.


## License

FBGEMM is BSD licensed, as found in the [`LICENSE`](LICENSE) file.
