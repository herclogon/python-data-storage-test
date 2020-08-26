# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem real type."""

import numpy

from .. import schema
from .common import SimpleHDFType


class Type(SimpleHDFType):
    """Real type implementation."""

    NAME = "Real"
    NATIVE_TYPE = float
    _STORAGE_DTYPE = (numpy.float64,)

    _SCHEMA = {
        schema.SchemaKeys.INIT: {"type": "number", "default": 0.0},
        "@nanable": {"type": "boolean", "default": True},
    }
    SCHEMA = schema.BOUNDED_SCHEMA("number", _SCHEMA)

    @classmethod
    def validate(cls, value, schema, errors_list):
        super(Type, cls).validate(value, schema, errors_list)
        if numpy.isnan(value):
            if not schema.get("@nanable", cls.SCHEMA["@nanable"]["default"]):
                errors_list.append(ValueError("NaN is not allowed"))
        else:
            cls._validate_value_bounds(value, schema, errors_list)

    @classmethod
    def to_hdf_column(cls, data, column_properties):
        return data.astype(cls._STORAGE_DTYPE[0]), cls._STORAGE_DTYPE[0]

    @classmethod
    def from_hdf_column(cls, data, column_properties):
        return data.astype(cls._STORAGE_DTYPE[0])
