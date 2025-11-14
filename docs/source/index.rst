.. librosax documentation master file

Librosax documentation
======================

**Librosax** is a JAX implementation of audio processing functions from `librosa <https://github.com/librosa/librosa/>`_
, `TorchLibrosa <https://github.com/qiuqiangkong/torchlibrosa>`_
, and `nnAudio <https://github.com/KinWaiCheuk/nnAudio>`_.

The source code is available on `GitHub <https://github.com/DBraun/librosax>`_.
Librosax follows `Effort-based Versioning <https://jacobtomlinson.dev/effver/>`_.

Getting started
---------------

.. toctree::
   :maxdepth: 1

   install
   changelog


API documentation
-----------------

.. toctree::
   :maxdepth: 2

   librosax
   feature
   layers


Citation
--------

If you use librosax in your research, please cite:

.. code-block:: bibtex

   @software{Braun_librosax_2025,
      author = {Braun, David},
      month = mar,
      title = {{librosax}},
      url = {https://github.com/DBraun/librosax},
      version = {0.0.5},
      year = {2025}
   }

Additionally, please consider citing the original libraries that librosax is based on:

**librosa** - For the design principles and most algorithms:

.. code-block:: bibtex

   @inproceedings{mcfee2015librosa,
     title={librosa: Audio and music signal analysis in python},
     author={McFee, Brian and Raffel, Colin and Liang, Dawen and Ellis, Daniel PW and McVicar, Matt and Battenberg, Eric and Nieto, Oriol},
     booktitle={Proceedings of the 14th python in science conference},
     volume={8},
     year={2015}
   }

**nnAudio** - For Constant-Q Transform (CQT) implementations:

.. code-block:: bibtex

   @ARTICLE{9174990,
     author={K. W. {Cheuk} and H. {Anderson} and K. {Agres} and D. {Herremans}},
     journal={IEEE Access},
     title={nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks},
     year={2020},
     volume={8},
     pages={161981-162003},
     doi={10.1109/ACCESS.2020.3019084}
   }

**TorchLibrosa** - For augmentations and neural network layers:

.. code-block:: bibtex

   @article{kong2020panns,
     title={{PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition}},
     author={Kong, Qiuqiang and Cao, Yin and Iqbal, Turab and Wang, Yuxuan and Wang, Wenwu and Plumbley, Mark D.},
     journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
     volume={28},
     pages={2880--2894},
     year={2020},
     publisher={{IEEE}}
   }


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`