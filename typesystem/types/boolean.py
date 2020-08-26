# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem boolean type."""

import numpy

from .. import schema
from .common import SimpleHDFType


class Type(SimpleHDFType):
    """Boolean type implementation."""

    NAME = "Boolean"
    NATIVE_TYPE = bool
    _STORAGE_DTYPE = (numpy.int8,)

    SCHEMA = {
        schema.SchemaKeys.INIT: {"type": "boolean", "default": False},
        "@true_str": {"type": "string", "default": "True"},
        "@false_str": {"type": "string", "default": "False"},
    }

    @classmethod
    def to_hdf_column(cls, data, column_properties):
        return data, cls._STORAGE_DTYPE[0]

    @classmethod
    def from_hdf_column(cls, data, column_properties):
        return data.astype(numpy.bool)
