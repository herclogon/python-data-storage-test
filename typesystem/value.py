# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem value.

Main class for storing data and properties, used for data serialization
and validation.
"""

import collections
import copy
import datetime
import json
import uuid

import jsonschema
import numpy

from . import constants, schema, types
from .utils import ErrorsList


class Value:
    """Value object, combined value and properties together, allows lazy
    loading, serialization and validation.

    When created from File lazy loading is enabled (if
    supported by underlying type). Native value and hdf_file can not be
    specified together.

    Args:
        value: Native python value.
        value_properties: `value` properties.
        hdf_file: opened h5py.File instance.
        hdf_path: (str) path inside HDF5 file
    """

    NAME = "Value"
    VALUE_PROPERTIES = {}

    @classmethod
    def update_properties(cls, value, value_properties):
        return value_properties

    @classmethod
    def compress_properties(cls, value_properties):
        return value_properties

    @staticmethod
    def update_schemas(schemas):
        return schemas

    @staticmethod
    def compress_schemas(schemas):
        return schemas

    @classmethod
    def to_native(cls, value, properties):
        return value

    @classmethod
    def from_native(cls, value, properties):
        return value

    def __init__(self, value=None, value_properties=None, hdf_file=None, hdf_path=None):
        value_properties = value_properties or {}
        assert isinstance(value_properties, collections.Mapping)
        assert (hdf_file is None) == (
            hdf_path is None
        ), "hdf_file and hdf_path must be set together!"

        self._hdf_file = hdf_file
        self._hdf_path = hdf_path

        if hdf_file is not None:
            node = hdf_file[hdf_path]
            value_type = node.attrs["@type"]
            from .constants import TYPES

            self._type = TYPES[value_type]
            read_properties = json.loads(node.attrs["@properties"])
            self._properties = self._type.update_properties(None, read_properties)
            self._data = self._type.read(hdf_file, hdf_path, self._properties)
        else:
            self._properties = copy.deepcopy(value_properties)
            self._type = resolve_type(value, self.properties)
            self._data = self._type.from_native(value, self.properties)

        self._update_and_validate_properties()

    def __repr__(self):
        # @TODO switch to reprlib
        return "Value({!r})".format(self.native)

    @property
    def type(self):
        """Value type as type class."""
        return self._type

    @property
    def native(self):
        """Native python value."""
        return self._type.to_native(self._data, self.properties)

    @native.setter
    def native(self, value):
        """Change value without changing type and properties.

        Args:
            value: New native value.

        """
        if resolve_type(value, self.properties) != self._type:
            raise NotImplementedError
        if self._hdf_file is not None:
            if not self._hdf_file:
                raise ValueError("File is closed")
            attr_type = self._hdf_file[self._hdf_path].attrs["@type"]
            attr_props = json.dumps(self.type.compress_properties(self.properties))
            del self._hdf_file[self._hdf_path]

            # NOTE: here the new Value won't be bound with the hdf-file
            #       so changing its subvalues won't affect the file.
            self._data = self._type.from_native(value, self.properties)
            self._type.write(
                self._hdf_file, self._hdf_path, self._data, self.properties
            )
            self._hdf_file[self._hdf_path].attrs["@type"] = attr_type
            self._hdf_file[self._hdf_path].attrs["@properties"] = attr_props

        else:
            self._data = self._type.from_native(value, self.properties)

    @property
    def data(self):
        """Get type-specific value object with lazy-loading if supported
        by type (may be the same as native value).
        """
        return self._data

    @property
    def properties(self):
        """Value properties."""
        return self._properties

    @properties.setter
    def properties(self, value: dict):
        """Set new properties."""
        self._properties = value

    def validate(self, value_properties=None, raise_on_error=True):
        """Validate value data against properties.

        Args:
            value_properties: Properties to validate against, if `None`
                own properties are used.
            raise_on_error: Raise error if any or store all errors in
                list and return them.

        Returns:
            Errors list if `raise_on_error` is `False` otherwise throws
                error.

        """
        errors = ErrorsList(raise_on_error)
        self._validate(value_properties, errors)
        return errors.list()

    def _validate(self, value_properties, errors_list):
        value_schema = self._properties[schema.PropertiesKeys.SCHEMA]
        assert value_schema[schema.SchemaKeys.TYPE] == self._type.NAME
        if value_properties is None:
            schemas = [value_schema]
            value_schemas = (
                self._properties.get(schema.PropertiesKeys.SCHEMAS, None) or schemas
            )
            if schemas[0] not in value_schemas:
                errors_list.append(
                    TypeError("Value schema not found in value schemas list!")
                )
        else:
            schemas = value_properties.get(schema.PropertiesKeys.SCHEMAS, None)
            if schemas is None:
                schemas = [value_properties[schema.PropertiesKeys.SCHEMA]]
            schemas = [
                s for s in schemas if s[schema.SchemaKeys.TYPE] == self._type.NAME
            ]
            if not schemas:
                errors_list.append(
                    TypeError("Value type does not match any schema in properties!")
                )

        validation_results = []
        for s in schemas:
            schema_errors = ErrorsList(False)
            self._type.validate(self._data, s, schema_errors)
            if not schema_errors.list():
                validation_results = []
                break
            else:
                validation_results += schema_errors.list()
        for error in validation_results:
            errors_list.append(error)

    def write(self, hdf_file, hdf_path=None):
        """Write value to the file.

        Args:
            file: File to write opened in binary mode and correct
                position must be set.

        """
        if not hdf_path:
            hdf_path = f"/{uuid.uuid4()}"
        self.type.write(hdf_file, hdf_path, self.data, self.properties)
        node = hdf_file[hdf_path]
        node.attrs["@type"] = self.type.NAME
        properties_to_write = self.type.compress_properties(self.properties)
        node.attrs["@properties"] = json.dumps(properties_to_write)

    def _update_and_validate_properties(self):
        self._properties = self._type.update_properties(self._data, self.properties)
        if self._type is not Value:
            # validate only non Value types
            jsonschema.validate(self.properties, schema.PROPERTIES_SCHEMA(self._type))


def is_convertable_to_matrix(list_value):
    """Check if a list value may be converted to a matrix."""

    # Scalar types suitable for storing in matrix.
    MATRIX_ITEM_TYPES = (types.integer.Type, types.real.Type)

    def check_items_in_a_row(row):
        """Check whether all elements in row are of suitable type."""
        return all(resolve_type(item) in MATRIX_ITEM_TYPES for item in row)

    if not list_value:
        return True

    # Assume input is a list of scalars.
    if not isinstance(list_value[0], (list, tuple)):
        return check_items_in_a_row(list_value)

    # If input is a list of lists check rectangular shape first.
    item_length = len(list_value[0])
    for item in list_value[1:]:
        if not isinstance(item, (list, tuple)) or len(item) != item_length:
            return False
    if item_length == 0:
        return True

    # Check that all items have suitable type.
    return all(map(check_items_in_a_row, list_value))


def resolve_type(value, value_properties=None):
    """Get type class for by value and properties.

    Args:
        value: Native value.
        value_properties: Value properties.

    Return:
        one of types from types module or (special case) Value


    Raises:
        ValueError - if value's type is not resolved
        NotImplementedError - if value and value_properties mismatch

    """
    # We nave many types and dividing type checks is meaningless.
    #  pylint: disable=too-many-branches

    INT_TYPES = (int, numpy.int_, numpy.uint8)
    FLOAT_TYPES = (float, numpy.float_)

    value_properties = value_properties or {}
    value_schema = value_properties.get(schema.PropertiesKeys.SCHEMA, {})
    schema_type = value_schema.get(schema.SchemaKeys.TYPE)

    if value is None:
        type_name = types.null.Type.NAME
    elif isinstance(value, (bytes, bytearray)):
        type_name = types.binary.Type.NAME
    elif isinstance(value, (bool, numpy.bool_)):
        type_name = types.boolean.Type.NAME
    elif isinstance(value, numpy.ndarray):
        type_name = types.table.Type.NAME
        if value.dtype.names is None or not value.dtype.names:
            if any(
                map(
                    lambda t: numpy.issubdtype(value.dtype, numpy.dtype(t)),
                    [numpy.int, numpy.float, numpy.uint8, numpy.uint],
                )
            ):
                type_name = types.matrix.Type.NAME
    elif isinstance(value, collections.Mapping):
        type_name = types.dictionary.Type.NAME
        if schema_type == types.structure.Type.NAME:
            type_name = schema_type
    elif isinstance(value, INT_TYPES):
        type_name = types.integer.Type.NAME
        if schema_type == types.real.Type.NAME:
            type_name = schema_type
    elif isinstance(value, FLOAT_TYPES):
        type_name = types.real.Type.NAME
    elif isinstance(value, slice):
        type_name = types.slice.Type.NAME
    elif isinstance(value, str):
        type_name = types.string.Type.NAME
        if schema_type in [
            types.enumeration.Type.NAME,
            types.path.Type.NAME,
            types.timestamp.Type.NAME,
        ]:
            type_name = schema_type
    elif isinstance(value, datetime.datetime):
        type_name = types.timestamp.Type.NAME
    elif isinstance(value, collections.Sequence):
        type_name = types.list.Type.NAME
        if schema_type == types.subset.Type.NAME:
            type_name = schema_type
        elif schema_type == types.matrix.Type.NAME and is_convertable_to_matrix(value):
            type_name = schema_type
    elif isinstance(value, Value):
        # special case
        return Value
    else:
        raise ValueError(f"Unresolved type '{type(value)}'!")

    if schema_type is not None and type_name != schema_type:
        if type_name == types.null.Type.NAME:
            if value_schema.get(schema.SchemaKeys.NULLABLE, False):
                return constants.TYPES[schema_type]
        raise NotImplementedError(
            f"Value type and properties mismatch:"
            f" value: {value},"
            f" type of value: {type_name},"
            f" selected type: {schema_type}!"
        )
    return constants.TYPES[type_name]
