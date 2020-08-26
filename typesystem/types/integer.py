# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem integer type."""

import numpy

from .. import schema
from .common import SimpleHDFType


class Type(SimpleHDFType):
    """Integer type implementation."""

    NAME = "Integer"
    NATIVE_TYPE = int
    _STORAGE_DTYPE = (numpy.int64,)

    _SCHEMA = {schema.SchemaKeys.INIT: {"type": "integer", "default": 0}}
    SCHEMA = schema.BOUNDED_SCHEMA("integer", _SCHEMA)

    @classmethod
    def validate(cls, value, schema, errors_list):
        super(Type, cls).validate(value, schema, errors_list)
        cls._validate_value_bounds(value, schema, errors_list)

    @classmethod
    def to_hdf_column(cls, data, column_properties):
        return data, cls._STORAGE_DTYPE[0]

    @classmethod
    def from_hdf_column(cls, data, column_properties):
        return data.astype(cls._STORAGE_DTYPE[0])
