# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem matrix type.

TODO:
  * add ability to store any shape, use '@shape' at value metadata
    and store matrix data as 8 columns table
"""

import numpy

from . import integer, real
from .. import schema
from .common import TypeBase


class Type(TypeBase):
    """Matrix type implementation."""

    NAME = "Matrix"
    SCHEMA = {
        schema.SchemaKeys.INIT: {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number", "default": 0.0},
                "default": [],
            },
            "default": [],
        },
        "@items": {
            "oneOf": [
                schema.PROPERTIES_DEFINITIONS_TYPE_REF(type.Type)
                for type in [real, integer]
            ]
        },
        "@nature": {"type": "string", "enum": ["square", "symmetric"]},
        "@allow_missing_items": {"type": "boolean", "default": True},
    }
    SCHEMA_REQUIRED = ["@items"]

    VALUE_PROPERTIES = {
        "@shape": {
            "type": "array",
            "items": {"type": "number", "default": 0},
            "minItems": 2,
            "maxItems": 2,
            "default": [0, 0],
        }
    }

    @classmethod
    def update_properties(cls, value, value_properties):
        result = super(Type, cls).update_properties(value, value_properties)
        value_schema = result[schema.PropertiesKeys.SCHEMA]
        if "@items" not in value_schema:
            items_type = real.Type
            if value is not None:
                if not numpy.issubdtype(value.dtype, numpy.dtype(numpy.float)):
                    items_type = integer.Type
            value_schema["@items"] = {schema.SchemaKeys.TYPE: items_type.NAME}
        value_schema["@items"] = Type.update_schemas([value_schema["@items"]])[0]

        if value is None:
            if "@shape" not in result:
                result["@shape"] = cls.VALUE_PROPERTIES["@shape"]["default"]
        else:
            result["@shape"] = list(value.shape)
        return result

    @classmethod
    def compress_properties(cls, value_properties):
        result = super(Type, cls).compress_properties(value_properties)
        value_schema = result.get(schema.PropertiesKeys.SCHEMA, {})
        value_schema["@items"] = Type.compress_schemas([value_schema["@items"]])[0]
        if 0 not in result["@shape"]:
            del result["@shape"]
        return result

    @classmethod
    def validate(cls, value, value_schema, errors_list):
        super(Type, cls).validate(value, value_schema, errors_list)
        allow_missing_default = cls.SCHEMA["@allow_missing_items"]["default"]
        allow_missing = value_schema.get("@allow_missing_items", allow_missing_default)
        if not allow_missing and numpy.any(value.mask):
            errors_list.append(ValueError("missing items in matrix"))

        nature = value_schema.get("@nature")
        if nature in ("square", "symmetric") and value.shape[0] != value.shape[1]:
            errors_list.append(ValueError("matrix is not square"))
        if nature == "symmetric":
            result = [False]
            if value.shape[0] == value.shape[1]:
                transposed = value.transpose()
                result = value == transposed
                for row in range(value.shape[0]):
                    for col in range(value.shape[1]):
                        if (
                            not result[row][col]
                            and numpy.isnan(value[row][col])
                            and numpy.isnan(transposed[col][row])
                        ):
                            result[row][col] = True
            if not numpy.all(result):
                errors_list.append(ValueError("matrix is not symmetric"))

        items_schema = value_schema["@items"]
        items_type = items_schema[schema.SchemaKeys.TYPE]
        if items_type == real.Type.NAME:
            items_type = real.Type
        elif items_type == integer.Type.NAME:
            items_type = integer.Type
        else:
            errors_list.append(ValueError(f"Unsupported items type: {items_type}!"))
            return

        for row in range(value.shape[0]):
            for col in range(value.shape[1]):
                if not value.mask[row][col]:
                    items_type.validate(value[row][col], items_schema, errors_list)

    @classmethod
    def read(cls, hdf_file, hdf_path, properties):
        node = hdf_file[hdf_path]
        if "mask" in node:
            mask = node["mask"][()]
        else:
            mask = False
        data = node["data"][()]
        return numpy.ma.masked_array(data, mask)

    @classmethod
    def write(cls, hdf_file, hdf_path, value, properties):
        node = hdf_file.create_group(hdf_path)
        if value.mask.any():
            node.create_dataset("mask", data=value.mask)
        node.create_dataset("data", data=value.data)

    @classmethod
    def from_native(cls, value, properties):
        from typesystem.value import is_convertable_to_matrix

        if not isinstance(value, numpy.ndarray):
            if not is_convertable_to_matrix(value):
                raise ValueError("Cannot convert value to the matrix!")
            value = numpy.array(value)

        if len(value.shape) != 2:
            value = value.reshape((-1, 1))

        if not hasattr(value, "mask"):
            mask = numpy.zeros(shape=value.shape, dtype=numpy.bool)
            return numpy.ma.masked_array(value, mask)
        return numpy.ma.masked_array(value)

    @classmethod
    def to_native(cls, value, properties):
        return value
