# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem null type."""

import numpy

from .. import schema
from .common import SimpleHDFType


class Type(SimpleHDFType):
    """Null type implementation."""

    NAME = "NULL"
    NATIVE_TYPE = None
    _STORAGE_DTYPE = ()

    SCHEMA = {schema.SchemaKeys.INIT: {"type": "null", "default": None}}

    @classmethod
    def raw_to_native(cls, value, properties):
        return cls.NATIVE_TYPE

    @classmethod
    def native_to_raw(cls, value, properties):
        return []

    @classmethod
    def to_hdf_column(cls, data, column_properties):
        data = numpy.array(
            [cls.native_to_raw(item, column_properties) for item in data]
        )
        return data, numpy.dtype(numpy.int8)

    @classmethod
    def from_hdf_column(cls, data, column_properties):
        return numpy.array([None] * data.shape[0], dtype=numpy.object)
