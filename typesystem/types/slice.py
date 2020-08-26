# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem slice type."""

import numpy

from .. import schema
from .common import SimpleHDFType


class Type(SimpleHDFType):
    """Slice type implementation."""

    NAME = "Slice"
    NATIVE_TYPE = slice
    _STORAGE_DTYPE = (numpy.int64, numpy.int64, numpy.int64)

    _SCHEMA = {
        schema.SchemaKeys.INIT: {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3,
            "default": [0, 0, 0],
        }
    }
    SCHEMA = schema.BOUNDED_SCHEMA("integer", _SCHEMA)

    @classmethod
    def validate(cls, value, schema, errors_list):
        super(Type, cls).validate(value, schema, errors_list)
        cls._validate_value_bounds(value.start, schema, errors_list)
        cls._validate_value_bounds(
            value.stop - ((value.stop - value.start) % value.step), schema, errors_list
        )

    @classmethod
    def from_native(cls, value, properties):
        if value.stop is None:
            raise ValueError("undefined iteration stop value")
        return slice(
            0 if value.start is None else value.start,
            value.stop,
            1 if value.step is None else value.step,
        )

    @classmethod
    def native_to_raw(cls, value, properties):
        if value.stop is None:
            raise ValueError("unsupported slice stop value")
        return [
            numpy.int64(value.start),
            numpy.int64(value.stop),
            numpy.int64(value.step),
        ]

    @classmethod
    def from_hdf_column(cls, data, column_properties):
        return numpy.array(
            [cls.raw_to_native(item, column_properties) for item in data],
            dtype=numpy.object,
        )
