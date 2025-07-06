#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Exception classes for librosa"""


class LibrosaxError(Exception):
    """The root librosa exception class"""

    pass


class ParameterError(LibrosaxError):
    """Exception class for mal-formed inputs"""

    pass