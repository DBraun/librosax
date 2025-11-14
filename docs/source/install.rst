.. _install:

Installation
============

Installing JAX (Optional but Recommended)
------------------------------------------

Although optional, we recommend first installing the `jax-ai-stack <https://github.com/jax-ml/jax-ai-stack>`_ with one of these three options:

.. code-block:: bash

   pip install jax-ai-stack              # JAX CPU
   pip install jax-ai-stack "jax[cuda]"  # JAX + AI stack with GPU/CUDA support
   pip install jax-ai-stack "jax[tpu]"   # JAX + AI stack with TPU support

For more specific JAX installation options, please refer to the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html#installation>`_.

Installing librosax
-------------------

**Required:** Then install librosax from PyPI:

.. code-block:: bash

   pip install librosax

Installing from Source
----------------------

To build librosax from source:

.. code-block:: bash

   git clone https://github.com/DBraun/librosax.git
   cd librosax
   pip install -e .

For development work, you can install additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"      # Install with testing dependencies
   pip install -e ".[docs]"     # Install with documentation dependencies
   pip install -e ".[dev,docs]" # Install with both

Dependencies
------------

Librosax automatically installs its required dependencies (JAX, Flax, librosa, NumPy, SciPy, and others) when you install it via pip. For the complete list of dependencies and their version requirements, see the `setup.cfg <https://github.com/DBraun/librosax/blob/main/setup.cfg>`_ file in the repository.