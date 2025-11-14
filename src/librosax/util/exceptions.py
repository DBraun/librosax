#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Exception classes for librosa"""


class LibrosaxError(Exception):
    """The root librosa exception class.
    
    All librosa-specific exceptions inherit from this base class.
    """

    pass


class ParameterError(LibrosaxError):
    """Exception class for mal-formed inputs.
    
    Raised when function parameters are invalid or incorrectly formatted.
    """

    pass