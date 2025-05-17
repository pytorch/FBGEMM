FBGEMM Releases
===============

.. _fbgemm.releases.compatibility:

FBGEMM Releases Compatibility Table
-----------------------------------

FBGEMM is released in accordance to the PyTorch release schedule, and is each
release has no guarantee to work in conjunction with PyTorch releases that are
older than the one that the FBGEMM release corresponds to.

+-----------------+------------------+------------------+----------------+--------------------+---------------------------+---------------------------+
| FBGEMM Release  | Corresponding    | Supported        | Supported      | Supported CUDA     | (Experimental) Supported  | (Experimental) Supported  |
|                 | PyTorch Release  | Python Versions  | CUDA Versions  | Architectures      | ROCm Versions             | ROCm Architectures        |
+=================+==================+==================+================+====================+===========================+===========================+
| 1.2.0           | 2.7.x            | 3.9, 3.10, 3.11, | 11.8, 12.6,    | 7.0, 8.0, 9.0,     | 6.2.4, 6.3                | gfx908, gfx90a, gfx942    |
|                 |                  | 3.12, 3.13       | 12.8           | 9.0a, 10.0a, 12.0a |                           |                           |
+-----------------+------------------+------------------+----------------+----------------------+---------------------------+---------------------------+
| 1.1.0           | 2.6.x            | 3.9, 3.10, 3.11, | 11.8, 12.4,    | 7.0, 8.0, 9.0,     | 6.1, 6.2.4, 6.3           | gfx908, gfx90a, gfx942    |
|                 |                  | 3.12, 3.13       | 12.6           | 9.0a               |                           |                           |
+-----------------+------------------+------------------+----------------+--------------------+---------------------------+---------------------------+
| 1.0.0           | 2.5.x            | 3.9, 3.10, 3.11, | 11.8, 12.1,    | 7.0, 8.0, 9.0,     | 6.0, 6.1                  | gfx908, gfx90a            |
|                 |                  | 3.12             | 12.4           | 9.0a               |                           |                           |
+-----------------+------------------+------------------+----------------+--------------------+---------------------------+---------------------------+
| 0.8.0           | 2.4.x            | 3.8, 3.9, 3.10,  | 11.8, 12.1,    | 7.0, 8.0, 9.0,     | 6.0, 6.1                  | gfx908, gfx90a            |
|                 |                  | 3.11, 3.12       | 12.4           | 9.0a               |                           |                           |
+-----------------+------------------+------------------+----------------+--------------------+---------------------------+---------------------------+
| 0.7.0           | 2.3.x            | 3.8, 3.9, 3.10,  | 11.8, 12.1     | 7.0, 8.0, 9.0      | 6.0                       | gfx908, gfx90a            |
|                 |                  | 3.11, 3.12       |                |                    |                           |                           |
+-----------------+------------------+------------------+----------------+--------------------+---------------------------+---------------------------+
| 0.6.0           | 2.2.x            | 3.8, 3.9, 3.10,  | 11.8, 12.1     | 7.0, 8.0, 9.0      | 5.7                       | gfx90a                    |
|                 |                  | 3.11, 3.12       |                |                    |                           |                           |
+-----------------+------------------+------------------+----------------+--------------------+---------------------------+---------------------------+
| 0.5.0           | 2.1.x            | 3.8, 3.9, 3.10,  | 11.8, 12.1     | 7.0, 8.0, 9.0      | 5.5, 5.6                  | gfx90a                    |
|                 |                  | 3.11             |                |                    |                           |                           |
+-----------------+------------------+------------------+----------------+--------------------+---------------------------+---------------------------+
| 0.4.0           | 2.0.x            | 3.8, 3.9, 3.10   | 11.7, 11.8     | 7.0, 8.0           | 5.3, 5.4                  | gfx90a                    |
+-----------------+------------------+------------------+----------------+--------------------+---------------------------+---------------------------+

Note that the list of supported CUDA and ROCm architectures refer to the targets
support available in the default installation packages, and that building for
other architecures may be possible, but not guaranteed.

For more information, please visit:

- `FBGEMM Releases Page <https://github.com/pytorch/FBGEMM/releases>`_
- `CUDA Architectures <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`_
- `ROCm Architectures <https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html>`_

FBGEMM Release Notes
--------------------

- `FBGEMM v1.1.0 Release Notes <https://github.com/pytorch/FBGEMM/releases/tag/v1.1.0>`_
- `FBGEMM v1.0.0 Release Notes <https://github.com/pytorch/FBGEMM/releases/tag/v1.0.0>`_
- `FBGEMM v1.8.0 Release Notes <https://github.com/pytorch/FBGEMM/releases/tag/v0.8.0>`_
- `FBGEMM v1.7.0 Release Notes <https://github.com/pytorch/FBGEMM/releases/tag/v0.7.0>`_
- `FBGEMM v1.6.0 Release Notes <https://github.com/pytorch/FBGEMM/releases/tag/v0.6.0>`_
- `FBGEMM v1.5.0 Release Notes <https://github.com/pytorch/FBGEMM/releases/tag/v0.5.0>`_
- `FBGEMM v1.4.0 Release Notes <https://github.com/pytorch/FBGEMM/releases/tag/v0.4.0>`_
