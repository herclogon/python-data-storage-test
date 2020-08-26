# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem base classes for types implementation."""

import copy

import numpy

from .. import schema


class TypeBase:
    """Base class for type implementation.

    Describes all methods and properties of type class, have default
    implementation for some of them.

    Serialization and deserialization methods works with type-specific
    value, which may be python-native type or some specific object it is
    guaranteed that such object is used in all methods which needs value
    except `from_native` which takes python-native value and returns
    value-object instead.
    """

    def __new__(cls):
        raise UserWarning("This class is not intended to be instantiated!")

    # Unique type name, used as type identifier, should be non-empty
    # string in Camel case.
    NAME = None
    # Python native type for the value.
    NATIVE_TYPE = None

    # JSON schema of type properties, see properties specification in
    # README for details.
    # - SCHEMA: Type schema, dict with additional properties for schema
    #   as json schema object, each type should specify at least '@init'
    #   property.
    # - SCHEMA_REQUIRED: List of required properties in schema.
    # - VALUE_PROPERTIES: JSON schema specification of object properties
    #   added to the JSON schema for value properties validation.
    SCHEMA = None
    SCHEMA_REQUIRED = []
    VALUE_PROPERTIES = {}

    @classmethod
    def read(cls, hdf_file, hdf_path, properties):
        """Read value stored in specific memory area.

        Args:
            hdf_file: opened h5py.File instance.
            hdf_path: path to value in HDF5 file.
            properties: Value properties, stored separately from value.

        Returns:
            Value object which will be used for all further operations
            with value.

        """
        raise NotImplementedError

    @classmethod
    def write(cls, hdf_file, hdf_path, value, properties):
        """Write value to file.

        Args:
            hdf_file: opened h5py.File instance.
            hdf_path: path to value in HDF5 file.
            value: Value-object specific for a type.
            properties: Value properties, they should not be stored in a
                file but still useful.

        """
        raise NotImplementedError

    @classmethod
    def to_native(cls, value, properties):
        """Convert value object to native python value.

        Args:
            value: Value object.
            properties: Value properties.

        """
        del properties
        return value

    @classmethod
    def from_native(cls, value, properties):
        """Create value object from native python representation.

        Args:
            value: Native python value.
            properties: Value properties.

        """
        del properties
        return value

    @classmethod
    def validate(cls, value, validation_schema, errors_list):
        """Validate value.

        Must be called from `validate` method in derived class.

        Args:
            value: Value object.
            validation_schema: Schema for value validation.
            errors_list: Container for validation errors, all validation
                errors should be stored here via append method, see
                `utils.ErrorsList`.

        """
        del value, errors_list
        assert validation_schema[schema.SchemaKeys.TYPE] == cls.NAME

    @classmethod
    def schema(cls):
        """Json-schema for type schema validation."""
        result = {
            "type": "object",
            "properties": {
                schema.SchemaKeys.TYPE: {
                    "type": "string",
                    "enum": [cls.NAME],
                    "default": cls.NAME,
                    "description": "Type name in typesystem",
                },
                schema.SchemaKeys.NAME: {
                    "type": "string",
                    "default": cls.NAME,
                    "description": "Specific type name",
                },
                schema.SchemaKeys.NULLABLE: {
                    "type": "boolean",
                    "default": False,
                    "description": "If 'true' it is possible to assign null to value",
                },
            },
            "required": cls.SCHEMA_REQUIRED
            + [schema.SchemaKeys.TYPE, schema.SchemaKeys.NAME, schema.SchemaKeys.INIT],
            "default": {},
            "description": "%s type specific properties" % cls.NAME,
        }
        result["properties"].update(cls.SCHEMA)
        return result

    @classmethod
    def update_properties(cls, value, value_properties):
        """Complete properties (fill missing default values) after value
        is loaded or created.

        Derived class method must call base class method.

        Args:
            value: Value object.
            value_properties: Value properties (inout).

        Returns:
            `value_properties`.

        """
        del value
        if schema.PropertiesKeys.SCHEMA not in value_properties:
            value_properties[schema.PropertiesKeys.SCHEMA] = {
                schema.SchemaKeys.TYPE: cls.NAME
            }
        value_properties[schema.PropertiesKeys.SCHEMA] = cls._update_schema(
            value_properties[schema.PropertiesKeys.SCHEMA]
        )
        return value_properties

    @classmethod
    def compress_properties(cls, value_properties):
        """Remove values which can be deduced from storage type or value
        type before storing properties.

        Args:
            value_properties: Properties to compress.

        Returns:
            Compressed copy of `value_properties`.

        """
        value_properties = copy.deepcopy(value_properties)
        value_schema = cls._compress_schema(
            value_properties[schema.PropertiesKeys.SCHEMA]
        )
        if not value_schema:
            value_properties.pop(schema.PropertiesKeys.SCHEMA)
        return value_properties

    @staticmethod
    def update_schemas(schemas):
        """Fill non required fields in each schema in a list.

        Args:
            schemas: List of value schema, at least `@type` key must be
                in schema.

        Returns:
            New list of schema.

        """
        # pylint: disable=protected-access
        assert isinstance(schemas, list), "schemas must be a list of schema"
        return [
            TypeBase._item_type(single_schema)._update_schema(single_schema)
            for single_schema in schemas
        ]

    @staticmethod
    def compress_schemas(schemas):
        """Remove unnecessary fields (except `@type`) from each schema
        in list.

        Args:
            schemas: List of value schema.

        Returns:
            New list of schema.

        """
        # pylint: disable=protected-access
        assert isinstance(schemas, list), "schemas must be a list of schema"
        result = []
        for single_schema in schemas:
            single_type = TypeBase._item_type(single_schema)
            single_schema = single_type._compress_schema(single_schema)
            single_schema[schema.SchemaKeys.TYPE] = single_type.NAME
            result.append(single_schema)
        return result

    # Helper functions for storing/restoring schema collection
    # i.e. possible values for items

    @classmethod
    def _update_schema(cls, value_schema):
        """Fill missing values in schema.

        Args:
            value_schema: Schema for update (inout).

        Returns:
            `value_schema`.

        """
        if schema.SchemaKeys.TYPE not in value_schema:
            value_schema[schema.SchemaKeys.TYPE] = cls.NAME
        if schema.SchemaKeys.NAME not in value_schema:
            value_schema[schema.SchemaKeys.NAME] = cls.NAME
        if schema.SchemaKeys.INIT not in value_schema:
            # @TODO find a way to add properties defaults automatically
            init_schema = cls.schema()["properties"][schema.SchemaKeys.INIT]
            value_schema[schema.SchemaKeys.INIT] = init_schema["default"]
        if schema.SchemaKeys.NULLABLE not in value_schema:
            nullable_schema = cls.schema()["properties"][schema.SchemaKeys.NULLABLE]
            value_schema[schema.SchemaKeys.NULLABLE] = nullable_schema["default"]
        return value_schema

    @classmethod
    def _compress_schema(cls, value_schema):
        """Remove deducible fields from schema.

        Args:
            value_schema: Schema to compress.

        Returns:
            Compressed copy of `value_schema`.
        """
        value_schema = copy.deepcopy(value_schema)
        if value_schema[schema.SchemaKeys.NAME] == value_schema[schema.SchemaKeys.TYPE]:
            value_schema.pop(schema.SchemaKeys.NAME)
        if value_schema[schema.SchemaKeys.TYPE] == cls.NAME:
            value_schema.pop(schema.SchemaKeys.TYPE)

        init_schema = cls.schema()["properties"][schema.SchemaKeys.INIT]
        if value_schema[schema.SchemaKeys.INIT] == init_schema["default"]:
            value_schema.pop(schema.SchemaKeys.INIT)
        return value_schema

    @staticmethod
    def _item_type(item_schema):
        """Type class from item schema."""
        from .. import constants

        return constants.TYPES[item_schema[schema.SchemaKeys.TYPE]]

    # Helper functions for validation.

    @classmethod
    def _validate_string_length(cls, value, validation_schema, errors_list):
        if (
            schema.SchemaKeys.MIN_LENGTH in validation_schema
            and len(value) < validation_schema[schema.SchemaKeys.MIN_LENGTH]
        ):
            errors_list.append(
                ValueError(
                    "value length is lower than %d"
                    % validation_schema[schema.SchemaKeys.MIN_LENGTH]
                )
            )
        if (
            schema.SchemaKeys.MAX_LENGTH in validation_schema
            and len(value) > validation_schema[schema.SchemaKeys.MAX_LENGTH]
        ):
            errors_list.append(
                ValueError(
                    "value length is greater than %d"
                    % validation_schema[schema.SchemaKeys.MAX_LENGTH]
                )
            )

    @classmethod
    def _validate_collection_size(cls, value, validation_schema, errors_list):
        if (
            schema.SchemaKeys.MIN_ITEMS in validation_schema
            and len(value) < validation_schema[schema.SchemaKeys.MIN_ITEMS]
        ):
            errors_list.append(
                ValueError(
                    "number of items must not be lower than %d"
                    % validation_schema[schema.SchemaKeys.MIN_ITEMS]
                )
            )
        if (
            schema.SchemaKeys.MAX_ITEMS in validation_schema
            and len(value) > validation_schema[schema.SchemaKeys.MAX_ITEMS]
        ):
            errors_list.append(
                ValueError(
                    "number of items must not be higher than %d"
                    % validation_schema[schema.SchemaKeys.MAX_ITEMS]
                )
            )

    @classmethod
    def _validate_value_bounds(cls, value, validation_schema, errors_list):
        value_schema = cls.schema()
        value_properties = value_schema["properties"]
        minimum = validation_schema.get(schema.SchemaKeys.MINIMUM, None)
        exclusive_minimum = validation_schema.get(
            schema.SchemaKeys.EXCLUSIVE_MINIMUM,
            value_properties[schema.SchemaKeys.EXCLUSIVE_MINIMUM]["default"],
        )
        if minimum is not None and not (
            minimum < value if exclusive_minimum else minimum <= value
        ):
            errors_list.append(
                ValueError(
                    "%d violates %slower bound '%s'"
                    % (value, "strict " if exclusive_minimum else "", minimum)
                )
            )

        maximum = validation_schema.get(schema.SchemaKeys.MAXIMUM, None)
        exclusive_maximum = validation_schema.get(
            schema.SchemaKeys.EXCLUSIVE_MAXIMUM,
            value_properties[schema.SchemaKeys.EXCLUSIVE_MAXIMUM]["default"],
        )
        if maximum is not None and not (
            value < maximum if exclusive_maximum else value <= maximum
        ):
            errors_list.append(
                ValueError(
                    "%d violates %supper bound '%s'"
                    % (value, "strict " if exclusive_maximum else "", maximum)
                )
            )


def nativeTypeWithAnotherAsStorage(type):
    """Class factory for creating base class for types with uses another
    typesystem type methods for binary serialization, for example
    `Enumeration` uses `String` for type methods for binary
    serialization. Adds _TYPE property with the storage type class.
    """

    class TypeProxy(type.__bases__[0]):
        """Define serialization methods via `type` class."""

        NATIVE_TYPE = type.NATIVE_TYPE
        _TYPE = type

        @classmethod
        def read(cls, hdf_file, hdf_path, properties):
            return cls._TYPE.read(hdf_file, hdf_path, properties)

        @classmethod
        def write(cls, hdf_file, hdf_path, value, properties):
            return cls._TYPE.write(hdf_file, hdf_path, value, properties)

        @classmethod
        def to_native(cls, value, properties):
            return cls._TYPE.to_native(value, properties)

        @classmethod
        def from_native(cls, value, properties):
            return cls._TYPE.from_native(value, properties)

        @classmethod
        def to_hdf_column(cls, data, column_properties):
            return cls._TYPE.to_hdf_column(data, column_properties)

        @classmethod
        def from_hdf_column(cls, column_data, column_properties):
            return cls._TYPE.from_hdf_column(column_data, column_properties)

    return TypeProxy


class SimpleHDFType(TypeBase):
    """Types that can be stored in a single HDF5 dataset.

    These types use native python value as value object and
    store it's value via list of fixed types.
    """

    _STORAGE_DTYPE = None

    @classmethod
    def raw_to_native(cls, value, properties):
        """Convert list of raw values (types are same as in
        `_STORAGE_DTYPE` to native type) for deserialization.
        """
        del properties
        return cls.NATIVE_TYPE(*value)  # pylint: disable=not-callable

    @classmethod
    def native_to_raw(cls, value, properties):
        """Convert native type to the list of raw values
        (as `_STORAGE_DTYPE`) for serialization.
        """
        del properties
        if len(cls._STORAGE_DTYPE) != 1:
            raise NotImplementedError
        return [cls._STORAGE_DTYPE[0](value)]

    @classmethod
    def read(cls, hdf_file, hdf_path, properties):
        raw_value = []
        for i, t in enumerate(cls._STORAGE_DTYPE):
            value = hdf_file[hdf_path][i]
            raw_value.append(value)
        return cls.raw_to_native(raw_value, properties)

    @classmethod
    def write(cls, hdf_file, hdf_path, value, properties):

        assert all(
            map(lambda item: item == cls._STORAGE_DTYPE[0], cls._STORAGE_DTYPE)
        ), "different types in _STORAGE_DTYPE are not supported at the moment"

        data = cls.native_to_raw(value, properties)

        raw_data_len = len(cls._STORAGE_DTYPE)
        if raw_data_len:
            shape = (raw_data_len,)
            dtype = numpy.dtype(cls._STORAGE_DTYPE[0])
        else:
            dtype = numpy.dtype(numpy.int8)
            shape = (0,)
        hdf_file.create_dataset(hdf_path, shape, dtype=dtype, data=data)

    @classmethod
    def to_hdf_column(cls, data, column_properties):
        """Prepare data to be written to a HDF5 dataset.

        Args:
            data: data to be stored in HDF5
            column_properties: column properties

        Return:
            data, dtype - data and dtype to store in the HDF5 dataset

        """
        dtype = numpy.dtype(cls._STORAGE_DTYPE[0])
        data = numpy.array(
            [cls.native_to_raw(item, column_properties) for item in data]
        )
        if data.dtype != dtype:
            data = data.astype(dtype)
        return data, dtype

    @classmethod
    def from_hdf_column(cls, column_data, column_properties):
        """Transform data read from a HDF5 dataset.

        Args:
            column_data: column data as it is stored in HDF5
            column_properties: column properties

        Return:
            numpy.array, containing transformed (if necessary) data

        """
        del column_properties
        return column_data
