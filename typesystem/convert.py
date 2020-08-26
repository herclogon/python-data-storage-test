# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem types conversions.

only @ref convert function should be exported from this module
"""

import copy
from datetime import datetime

import numpy

from . import constants, types
from .schema import PropertiesKeys, SchemaKeys
from .value import Value


ISO_8601_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def convert(schema, value, value_properties=None):
    """ Convert value to specific schema,
        value may be native value or typesystem Value instance.
    Args:
        schema: destination type schema.
        value: value to convert, may be Value instance or native type.
        value_properties: if `value` is native used to create Value instance.
    Returns: Value instance if `value` is Value instance
        and native-type result if `value` is native type.
    """
    return_value_instance = isinstance(value, Value)
    if return_value_instance:
        if value_properties is not None:
            raise ValueError(
                "can not use value_properties argument along with value instance"
            )
    else:
        value = Value(value, value_properties or {})

    result = _convert(schema, value)
    assert (
        result.properties[PropertiesKeys.SCHEMA][SchemaKeys.TYPE]
        == schema[SchemaKeys.TYPE]
    )
    return result if return_value_instance else result.native


# converters table {source-name: {destination-name: converter-function}}
# converter function takes schema and Value as argument
# and returns Value of desired type as result
_CONVERTERS = {}


# convert Value instance to specific schema by choosing converter
# from predefined list of functions
def _convert(schema, value):
    assert isinstance(value, Value)
    type = schema[SchemaKeys.TYPE]
    converters = _CONVERTERS[value._type.NAME]
    if type not in converters:
        raise ValueError("no way to convert %s to %s" % (value._type.NAME, type))
    return converters[type](schema, value)


# create new properties with updated schema
def _update_schema(value_properties, schema):
    value_properties = copy.deepcopy(value_properties)
    value_properties[PropertiesKeys.SCHEMA] = schema
    return value_properties


# converters factory, all _make* functions return
# function(schema, Value) -> Value for using in _CONVERTERS


# create converter function from Value to native type converter
def _make_value_converter(converter):
    return lambda schema, value: Value(
        converter(value), _update_schema(value.properties, schema)
    )


# create converter function from native type to native type converter
def _make_native_converter(converter):
    return _make_value_converter(lambda x: converter(x.native))


# make converter for converting native single scalar with known dtype to matrix
def _make_scalar_to_matrix_converter(dtype):
    return _make_native_converter(lambda x: numpy.array([[x]], dtype=dtype))


# converter functions
def _convert_scalar_to_list(schema, value):
    return Value([value], _update_schema(value.properties, schema))


# converters from collection to scalar with possibility of recursive conversion
def _list_to_scalar(schema, value):
    if len(value.data) != 1:
        raise ValueError("list size must be exactly 1")
    return _convert(schema, value.data.value(0))


def _matrix_to_scalar(schema, value):
    if value.data.shape != (1, 1):
        raise ValueError("matrix size must be exactly 1x1")
    return _convert(schema, Value(value.data[0][0]))


def _table_to_scalar(schema, value):
    if len(value.data) != 1 or value.data.ncols != 1:
        raise ValueError("table size must be exactly 1x1")

    if isinstance(value.native, numpy.ma.core.MaskedArray):
        # workaround for what seems to be a bug in numpy
        return _convert(schema, Value(value.native[0][0]))
    return _convert(schema, Value(value.native[0].tolist()[0]))


def _string_to_boolean(schema, value):
    lower = value.native.lower()
    if "@true_str" in schema:
        if lower == schema["@true_str"].lower():
            return _convert(schema, Value(True))
    elif lower in ["true", "1", "yes"]:
        return _convert(schema, Value(True))
    if "@false_str" in schema:
        if lower == schema["@false_str"]:
            return _convert(schema, Value(False))
    elif lower in ["false", "0", "no"]:
        return _convert(schema, Value(False))
    raise ValueError("string is not convertible to boolean: '%s'" % value)


# value to native type converters,
# value is used in order to prevent loading whole data in memory
# or for getting additional properties


def _boolean_to_string(value):
    schema = value.properties[PropertiesKeys.SCHEMA]
    if value.native:
        return schema["@true_str"] if "@true_str" in schema else "True"
    return schema["@false_str"] if "@false_str" in schema else "False"


def _matrix_to_list(value):
    if len(value.data) != 1:
        raise ValueError("matrix must have exactly one row")
    return value.data[0].tolist()


def _table_to_dictionary(value):
    if len(value.data) != 1:
        raise ValueError("table must have exactly one row")
    data = value.native
    return {name: data[name].tolist()[0] for name in data.dtype.names}


def _table_to_list(value):
    assert isinstance(value, Value) and value._type.NAME == types.table.Type.NAME
    if len(value.data) == 1 or value.data.ncols == 1:
        data = value.native
        if len(value.data) == 1:
            return data[0].tolist()
        return data[data.dtype.names[0]].tolist()
    raise ValueError("table must have one row or one column")


def _table_to_matrix(value):
    column_types = [value.data.dtype[name] for name in value.data.dtype.names]
    convertible_mask = [False] * len(column_types)
    for dtype in [numpy.bool_, numpy.int64, numpy.float64]:
        convertible_mask = [
            convertible or numpy.issubdtype(type, dtype)
            for convertible, type in zip(convertible_mask, column_types)
        ]
        if all(convertible_mask):
            data = value.native
            matrix = numpy.zeros(shape=(len(data), value.data.ncols), dtype=dtype)
            for idx, name in enumerate(data.dtype.names):
                matrix[:, idx] = data[name]
            return numpy.ma.masked_array(
                matrix, mask=numpy.zeros(shape=matrix.shape, dtype=numpy.bool_)
            )
    raise ValueError("impossible to convert table to matrix, wrong column types")


# fill converters table, first default implementation for the same type and null
for name in constants.TYPES.keys():
    _CONVERTERS[name] = {
        name: _make_native_converter(lambda x: x),
        types.null.Type.NAME: _make_native_converter(lambda x: None),
    }

# add special converters
_CONVERTERS[types.binary.Type.NAME][types.matrix.Type.NAME] = _make_native_converter(
    lambda x: numpy.array([list(x)], dtype=numpy.uint8)
)

_CONVERTERS[types.boolean.Type.NAME][types.string.Type.NAME] = _make_value_converter(
    _boolean_to_string
)

_CONVERTERS[types.list.Type.NAME][types.matrix.Type.NAME] = _make_native_converter(
    lambda x: numpy.array([x])
)

_CONVERTERS[types.matrix.Type.NAME][types.list.Type.NAME] = _make_value_converter(
    _matrix_to_list
)

_CONVERTERS[types.string.Type.NAME][types.boolean.Type.NAME] = _string_to_boolean
_CONVERTERS[types.string.Type.NAME][types.timestamp.Type.NAME] = _make_native_converter(
    lambda x: datetime.strptime(x, ISO_8601_TIME_FORMAT)
)

_CONVERTERS[types.table.Type.NAME][types.dictionary.Type.NAME] = _make_value_converter(
    _table_to_dictionary
)
_CONVERTERS[types.table.Type.NAME][types.list.Type.NAME] = _make_value_converter(
    _table_to_list
)
_CONVERTERS[types.table.Type.NAME][types.matrix.Type.NAME] = _make_value_converter(
    _table_to_matrix
)
_CONVERTERS[types.table.Type.NAME][types.structure.Type.NAME] = _make_value_converter(
    _table_to_dictionary
)

_CONVERTERS[types.timestamp.Type.NAME][types.string.Type.NAME] = _make_native_converter(
    lambda x: x.strftime(ISO_8601_TIME_FORMAT)
)

# rules for native -> new_type(native) conversions,
# by list of pairs (source-type, [destination-type*])
_CONVERTIBLE_VIA_NATIVE_TYPE = [
    (types.binary, [types.list]),
    (types.boolean, [types.integer, types.real]),
    (types.dictionary, [types.structure]),
    (types.enumeration, [types.string]),
    (types.integer, [types.boolean, types.real, types.string]),
    (types.path, [types.string]),
    (types.real, [types.boolean, types.integer, types.string]),
    (types.string, [types.enumeration, types.integer, types.path, types.real]),
    (types.structure, [types.dictionary]),
    (types.subset, [types.list]),
]

for source, destinations in _CONVERTIBLE_VIA_NATIVE_TYPE:
    _CONVERTERS[source.Type.NAME].update(
        {
            type.Type.NAME: _make_native_converter(type.Type.NATIVE_TYPE)
            for type in destinations
        }
    )

# add scalar -> list of single scalar converters
_SCALARS_CONVERTIBLE_TO_LIST = [
    types.boolean,
    types.enumeration,
    types.integer,
    types.path,
    types.real,
    types.slice,
    types.string,
    types.timestamp,
]
for type in _SCALARS_CONVERTIBLE_TO_LIST:
    _CONVERTERS[type.Type.NAME][types.list.Type.NAME] = _convert_scalar_to_list

# add scalar -> matrix of single scalar converters (type, matrix-dtype)
_SCALARS_CONVERTIBLE_TO_MATRIX = [
    (types.boolean, numpy.uint8),
    (types.integer, numpy.int64),
    (types.real, numpy.float64),
]
for type, dtype in _SCALARS_CONVERTIBLE_TO_MATRIX:
    _CONVERTERS[type.Type.NAME][
        types.matrix.Type.NAME
    ] = _make_scalar_to_matrix_converter(dtype)

# add collection to scalar converters
_COLLECTION_TO_SCALAR_CONVERSION_RULES = [
    (
        types.list,
        _list_to_scalar,
        [
            types.boolean,
            types.enumeration,
            types.integer,
            types.real,
            types.slice,
            types.string,
            types.path,
            types.timestamp,
        ],
    ),
    (types.matrix, _matrix_to_scalar, [types.boolean, types.integer, types.real]),
    (
        types.table,
        _table_to_scalar,
        [
            types.boolean,
            types.enumeration,
            types.integer,
            types.path,
            types.real,
            types.slice,
            types.string,
            types.timestamp,
        ],
    ),
]
for source, converter, destinations in _COLLECTION_TO_SCALAR_CONVERSION_RULES:
    _CONVERTERS[source.Type.NAME].update(
        {type.Type.NAME: converter for type in destinations}
    )
