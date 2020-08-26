# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem list type."""

import collections

from . import dictionary
from .. import schema
from .common import TypeBase


class Type(TypeBase):
    """List type implementation.

    All functions expect instance of List object as value-object.
    """

    NAME = "List"
    NATIVE_TYPE = list

    _SCHEMA = {
        schema.SchemaKeys.INIT: {"type": "array", "default": []},
        schema.SchemaKeys.ITEMS: schema.PROPERTIES_DEFINITIONS_SCHEMAS_REF(),
        "@unique_items": {"type": "boolean", "default": False},
    }
    SCHEMA = schema.COLLECTION_SCHEMA(_SCHEMA)

    @classmethod
    def validate(cls, value, value_schema, errors_list):
        super(Type, cls).validate(value, value_schema, errors_list)
        cls._validate_collection_size(value, value_schema, errors_list)
        if "@items" in value_schema:
            schemas = {schema.PropertiesKeys.SCHEMAS: value_schema["@items"]}
            for idx in range(len(value)):
                value.value(idx)._validate(schemas, errors_list)

        if value_schema.get("@unique_items", cls._SCHEMA["@unique_items"]["default"]):
            unique = set()
            for idx, item in enumerate(value):
                if item in unique:
                    errors_list.append(ValueError("duplicated items in list"))
                    break
                unique.add(item)

    @classmethod
    def read(cls, hdf_file, hdf_path, properties):
        return List(None, hdf_file, hdf_path)

    @classmethod
    def write(cls, hdf_file, path, value, properties):
        g = hdf_file.create_group(path)
        for i in range(len(value)):
            value.value(i).write(hdf_file, f"{path}/{i}")

    @classmethod
    def to_native(cls, value, properties):
        return [item for idx, item in enumerate(value)]

    @classmethod
    def from_native(cls, value, properties):
        return List(value)

    @classmethod
    def update_properties(cls, value, value_properties):
        result = super(Type, cls).update_properties(value, value_properties)
        return dictionary.Type.update_items(result)

    @classmethod
    def compress_properties(cls, value_properties):
        result = super(Type, cls).compress_properties(value_properties)
        return dictionary.Type.compress_items(result)


class List(object):
    """Value object, uses dictionary.Dictionary as underlying storage
       since binary formats are the same.
    """

    def __init__(self, data, hdf_file=None, hdf_path=None):
        assert (hdf_file is None) == (hdf_path is None) and (data is None) != (
            hdf_file is None
        )
        assert data is None or isinstance(data, collections.Sequence)

        self._hdf_file = hdf_file
        self._hdf_path = hdf_path
        self._values = {}

        if hdf_file is None:
            from ..value import Value

            def box(value):
                if isinstance(value, Value):
                    return value
                return Value(value)

            self._values = {idx: box(item) for idx, item in enumerate(data)}
            return

        n_items = len(hdf_file[hdf_path])
        for i in range(n_items):
            assert f"{i}" in hdf_file[hdf_path], f"{i} item is missing"

    def value(self, idx):
        from ..value import Value

        if self._hdf_file is None:
            if idx not in self._values:
                raise IndexError
            return self._values[idx]

        if idx < 0 or idx >= len(self._hdf_file[self._hdf_path]):
            raise IndexError
        if idx in self._values:
            return self._values[idx]

        v = Value(hdf_file=self._hdf_file, hdf_path=f"{self._hdf_path}/{idx}")
        self._values[idx] = v
        return v

    def pop(self, idx):
        if idx not in self._values and self._hdf_file is None:
            raise IndexError

        if idx in self._values:
            self._values.pop(idx)
            for i in range(idx + 1, len(self) + 1):
                if i in self._values:
                    self._values[i - 1] = self._values[i]
                    del self._values[i]

        if self._hdf_file:
            self._hdf_file.pop(f"{self._hdf_path}/{idx}")
            for i in range(idx + 1, len(self) + 1):
                self._hdf_file.move(f"{self._hdf_path}/{i}", f"{self._hdf_path}/{i-1}")

    def __getitem__(self, idx):
        return self.value(idx).native

    def __setitem__(self, idx, value):
        self.value(idx).native = value

    def __delitem__(self, idx):
        self.pop(idx)

    def __len__(self):
        if self._hdf_file is None:
            return len(self._values)
        return len(self._hdf_file[self._hdf_path])
