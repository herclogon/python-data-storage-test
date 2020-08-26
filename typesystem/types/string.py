# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem string type."""

import re

import h5py
import numpy

from .. import schema
from .common import SimpleHDFType


class Type(SimpleHDFType):
    """String type implementation."""

    NAME = "String"
    NATIVE_TYPE = str
    ENCODING = "utf-8"

    _SCHEMA = {
        "@pattern": {"type": "string", "minLength": 1},
        "@format": {"type": "string", "minLength": 1},
        "@format_name": {"type": "string", "minLength": 1},
    }
    SCHEMA = schema.STRING_SCHEMA(_SCHEMA)

    @classmethod
    def validate(cls, value, schema, errors_list):
        super(Type, cls).validate(value, schema, errors_list)
        cls._validate_string_length(value, schema, errors_list)
        if "@pattern" in schema and not re.match(schema["@pattern"], value):
            errors_list.append(ValueError("value does not match specified pattern"))

    @classmethod
    def read(cls, hdf_file, hdf_path, properties):
        value = hdf_file[hdf_path][0]
        return value

    @classmethod
    def write(cls, hdf_file, path, value, properties):
        dtype = h5py.special_dtype(vlen=str)
        ds = hdf_file.create_dataset(path, (1,), dtype=dtype, data=value)

    @classmethod
    def to_hdf_column(cls, data, column_properties):
        dtype = h5py.special_dtype(vlen=str)
        if data.dtype != dtype:
            data = data.astype(dtype)
        return data, dtype

    @classmethod
    def from_hdf_column(cls, column_data, column_properties):
        del column_properties
        return column_data.astype(numpy.object)
