#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Exception classes for librosax"""

# Re-export librosa's exception classes for compatibility
# This ensures that exceptions raised by librosa functions wrapped by librosax
# can be caught using librosax exception types
from librosa.util.exceptions import LibrosaError as LibrosaxError
from librosa.util.exceptions import ParameterError

__all__ = ["LibrosaxError", "ParameterError"]