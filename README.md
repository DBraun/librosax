# librosax

A [JAX](https://docs.jax.dev/en/latest/)/[Flax](https://flax.readthedocs.io/en/latest/) implementation of audio processing functions, inspired by and building upon [librosa](https://github.com/librosa/librosa) and [TorchLibrosa](https://github.com/qiuqiangkong/torchlibrosa).

## Installation

Although, it is optional, we recommend first installing the [jax-ai-stack](https://github.com/jax-ml/jax-ai-stack) with one of these three options:

```bash
pip install jax-ai-stack              # JAX CPU
pip install jax-ai-stack "jax[cuda]"  # JAX + AI stack with GPU/CUDA support
pip install jax-ai-stack "jax[tpu]"   # JAX + AI stack with TPU support
```

**Required**: Then install librosax:

```bash
pip install librosax
```

## Documentation

Documentation is [here](http://dirt.design/librosax).

## Acknowledgments

This library is heavily inspired by and borrows code from:

- **[librosa](https://github.com/librosa/librosa)** - The excellent Python library for audio and music analysis by the librosa development team
- **[TorchLibrosa](https://github.com/qiuqiangkong/torchlibrosa)** - PyTorch implementations of librosa functions and neural net layers by Qiuqiang Kong
- **[nnAudio](https://github.com/KinWaiCheuk/nnAudio)** - PyTorch implementations of CQT and other functions by Kin Wai Cheuk

## License

librosax is licensed under the ISC License, matching the license used by librosa. See the [LICENSE](LICENSE) file for details.
