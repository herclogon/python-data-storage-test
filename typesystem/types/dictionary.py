# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem dictionary type."""

import collections
import uuid

from .. import schema
from ..value import Value
from .common import TypeBase


class Type(TypeBase):
    """Dictionary type implementation.

    All functions expect instance of Dictionary object as value-object.
    """

    NAME = "Dictionary"
    NATIVE_TYPE = dict

    _SCHEMA = {
        schema.SchemaKeys.INIT: {"type": "array", "default": []},
        schema.SchemaKeys.ITEMS: schema.PROPERTIES_DEFINITIONS_SCHEMAS_REF(),
    }
    SCHEMA = schema.COLLECTION_SCHEMA(_SCHEMA)

    @classmethod
    def read(cls, hdf_file, hdf_path, properties):
        return Dictionary(None, hdf_file=hdf_file, hdf_path=hdf_path)

    @classmethod
    def write(cls, hdf_file, hdf_path, value, properties):
        value.write(hdf_file, hdf_path)

    @classmethod
    def to_native(cls, value, properties):
        return {k: value[k] for k in value.keys()}

    @classmethod
    def from_native(cls, value, properties):
        return Dictionary(value)

    @staticmethod
    def update_items(value_properties):
        """Update `@items` member in schema."""
        value_schema = value_properties[schema.PropertiesKeys.SCHEMA]
        if schema.SchemaKeys.ITEMS in value_schema:
            value_schema[schema.SchemaKeys.ITEMS] = Type.update_schemas(
                value_schema[schema.SchemaKeys.ITEMS]
            )
        return value_properties

    @staticmethod
    def compress_items(value_properties):
        """Compress `@items` member in schema."""
        value_schema = value_properties.get(schema.PropertiesKeys.SCHEMA, {})
        if schema.SchemaKeys.ITEMS in value_schema:
            value_schema[schema.SchemaKeys.ITEMS] = Type.compress_schemas(
                value_schema[schema.SchemaKeys.ITEMS]
            )
        return value_properties

    @classmethod
    def update_properties(cls, value, value_properties):
        result = super(Type, cls).update_properties(value, value_properties)
        return Type.update_items(result)

    @classmethod
    def compress_properties(cls, value_properties):
        result = super(Type, cls).compress_properties(value_properties)
        return Type.compress_items(result)

    @classmethod
    def validate(cls, value, value_schema, errors_list):
        super(Type, cls).validate(value, value_schema, errors_list)
        cls._validate_collection_size(value, value_schema, errors_list)
        if "@items" in value_schema:
            schemas = {schema.PropertiesKeys.SCHEMAS: value_schema["@items"]}
            for k in value.keys():
                value.value(k)._validate(schemas, errors_list)


class Dictionary(object):
    """Value object proxy for lazy-loading."""

    def __init__(self, data, hdf_file=None, hdf_path=None):
        assert (hdf_file is None) == (hdf_path is None) and (data is None) != (
            hdf_file is None
        )
        assert data is None or isinstance(data, collections.Mapping)

        self._hdf_file = hdf_file
        self._hdf_path = hdf_path
        self._data = {}

        if hdf_file is None:
            for k in data.keys():
                if type(k) != str:
                    raise TypeError(f"Unsupported key type: {type(k)}")
            self._data = {
                k: v if isinstance(v, Value) else Value(v) for k, v in data.items()
            }
        else:
            node = hdf_file[hdf_path]
            self._data = {}

    def write(self, hdf_file, hdf_path):
        hdf_file.create_group(hdf_path)
        for key in self.keys():
            self.value(key).write(hdf_file, f"{hdf_path}/{key}")

    def value(self, key):
        key = str(key)
        from ..value import Value

        if key not in self:
            raise KeyError(key)
        if self._hdf_file is None:
            return self._data[key]
        if key not in self._data:
            self._data[key] = Value(
                hdf_file=self._hdf_file, hdf_path=f"{self._hdf_path}/{key}"
            )
        return self._data[key]

    def keys(self):
        if self._hdf_file is None:
            return self._data.keys()
        else:
            return self._hdf_file[self._hdf_path].keys()

    def pop(self, key):
        key = str(key)
        if key not in self:
            raise KeyError(key)

        if self._hdf_file is None:
            self._data.pop(key)
            return
        if not self._hdf_file:
            raise ValueError("File is closed")

        self._data.pop(key, None)
        node = self._hdf_file[self._hdf_path]
        del node[key]

    def __getitem__(self, key):
        return self.value(key).native

    def __setitem__(self, key, value):
        self.value(key).native = value

    def __delitem__(self, key):
        self.pop(key)

    def __contains__(self, key):
        return key in self.keys()

    def __len__(self):
        return len(self.keys())
