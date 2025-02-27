Feature Gates
=================

Feature Gates are a mechanism provided in the FBGEMM_GPU codebase that provides
a consistent way to enable and disable experimental features based on
environment settings.

While it can be thought of as a type-safe abstraction over environment
variables, note that feature gates are a **run-time mechanism** for controlling
code behavior.

Creating a Feature Gate
-------------------------------------

Feature Gates should be created if the intent is to land a feature into the
codebase, but defer its enablement until further verification in production
workloads.

C++
~~~

To define a feature gate on the C++ side, append to the
`ENUMERATE_ALL_FEATURE_FLAGS` X-macros definition in
`fbgemm_gpu/config/feature_gates.h <https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/include/fbgemm_gpu/config/feature_gates.h>`_:

.. code:: cpp

  #define ENUMERATE_ALL_FEATURE_FLAGS   \
    X(...)                              \
    ...                                 \
    X(EXAMPLE_FEATURE)  // <-- Append here

Python
~~~~~~

To define a feature gate on the Python side, simply add a new
value to the `FeatureGateName` enum definition in
`fbgemm_gpu/config/feature_list.py <https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/config/feature_list.py>`_:

.. code:: python

  class FeatureGateName(Enum):
    ...
    # Add here
    EXAMPLE_FEATURE = auto()

While not required, it is best to mirror the enum values defined in
`fbgemm_gpu/config/feature_gates.h` for consistency.


Enabling a Feature Gate
------------------------

See the documentation in :ref:`fbgemm-gpu.dev.config.cpp` and
:ref:`fbgemm-gpu.dev.config.python` for code examples of how to
enable feature gates.
