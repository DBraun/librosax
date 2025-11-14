.. _install:

Installation
============

The latest stable release is available on PyPI, and you can install it by running::

    pip install librosax

To build librosax from source, use::

    git clone https://github.com/DBraun/librosax.git
    cd librosax
    pip install -e .

Dependencies
------------

Librosax requires the following packages:

- jax-ai-stack >= 2025.2.5
- librosa >= 0.10.1
- numpy
- scipy
- einops

JAX Installation
----------------

For specific JAX installation options (CPU-only, GPU, TPU), please refer to the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html#installation>`_.