FBGEMM_GPU Stable Python API
============================

We provide the stable API support starting from FBGEMM_GPU v1.0. The following
outlines our supports:

- API backward compatibility guarantees via thorough testing. We guarantee that
  our stable APIs will be backward compatible within a major version, meaning
  that the stable APIs for v1.0.0 will be compatible with every future release
  unless explicitly announced in advance

- Enhanced documentation, ensuring that every stable API has comprehensive and
  up-to-date documentation.

- Functionality guarantees are only provided through unit testing framework.
  We do NOT guarantee any functionalities that are NOT explicitly tested and
  documented in our unit tests.

- No performance guarantees. However, we are committed to providing support on
  a best-effort basis.

Stable APIs
-----------

Our stable APIs can be found via the links below:

- Table batched embedding (TBE) modules (:ref:`training<tbe-ops-training-stable-api>` and :ref:`inference<tbe-ops-inference-stable-api>`)

- :ref:`Pooled embedding operators<pooled-embedding-operators-stable-api>`

- :ref:`Pooled embedding modules<pooled-embedding-modules-stable-api>`

- :ref:`Sparse operators<sparse-ops-stable-api>`

- :ref:`Jagged tensor operators<jagged-tensor-ops-stable-api>`

- :ref:`Quantization operators<quantize-ops-stable-api>`
