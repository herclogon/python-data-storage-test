# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem structure type."""

import collections

from . import dictionary
from .. import schema
from .common import TypeBase


class Type(TypeBase):
    """Structure type implementation."""

    NAME = "Structure"
    NATIVE_TYPE = dict

    SCHEMA = {
        schema.SchemaKeys.INIT: {"type": "array", "default": []},
        "@items": {
            "type": "object",
            "patternProperties": {"^.+$": schema.PROPERTIES_DEFINITIONS_SCHEMAS_REF()},
        },
        "@allow_missing_properties": {"type": "boolean", "default": True},
        "@allow_additional_properties": {"type": "boolean", "default": True},
    }

    @classmethod
    def update_properties(cls, value, value_properties):
        result = super(Type, cls).update_properties(value, value_properties)
        value_schema = result[schema.PropertiesKeys.SCHEMA]
        if "@items" in value_schema:
            value_schema["@items"] = {
                key: Type.update_schemas(schemas)
                for key, schemas in value_schema["@items"].items()
            }
        return result

    @classmethod
    def compress_properties(cls, value_properties):
        result = super(Type, cls).compress_properties(value_properties)
        value_schema = result.get(schema.PropertiesKeys.SCHEMA, {})
        if "@items" in value_schema:
            value_schema["@items"] = {
                key: Type.compress_schemas(schemas)
                for key, schemas in value_schema["@items"].items()
            }
        return result

    @classmethod
    def to_native(cls, value, properties):
        return {k: value[k] for k in value.keys()}

    @classmethod
    def from_native(cls, value, properties):
        return Structure(value)

    @classmethod
    def read(cls, hdf_file, hdf_path, properties):
        return Structure(None, hdf_file, hdf_path)

    @classmethod
    def write(cls, hdf_file, hdf_path, value, properties):
        return value.write(hdf_file, hdf_path)

    @classmethod
    def validate(cls, value, value_schema, errors_list):
        from ..value import Value

        super(Type, cls).validate(value, value_schema, errors_list)

        items_schema = value_schema.get("@items") or {}
        allow = value_schema.get(
            "@allow_additional_properties",
            cls.SCHEMA["@allow_additional_properties"]["default"],
        )
        if not allow and len(set(value.keys()) - set(items_schema.keys())):
            errors_list.append(ValueError("additional properties are not allowed"))

        allow = value_schema.get(
            "@allow_missing_properties",
            cls.SCHEMA["@allow_missing_properties"]["default"],
        )
        if not allow and len(set(items_schema.keys()) - set(value.keys())):
            errors_list.append(ValueError("missing properties are not allowed"))
        for property, property_schemas in items_schema.items():
            if property in value:
                Value(value[property])._validate(
                    {schema.PropertiesKeys.SCHEMAS: property_schemas}, errors_list
                )


class Structure(dictionary.Dictionary):
    """Structure value object proxy.

    No lazy loading (on the top level)
    """

    def __init__(self, data, hdf_file=None, hdf_path=None):
        super().__init__(data, hdf_file, hdf_path)
        from ..value import Value

        if self._hdf_file is not None:
            # load all the values
            node = self._hdf_file[self._hdf_path]
            for key in node.keys():
                self._data[key] = Value(
                    hdf_file=self._hdf_file, hdf_path=f"{self._hdf_path}/{key}"
                )

    def value(self, key):
        key = str(key)
        if key not in self:
            raise KeyError(key)
        return self._data[key]
