# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem json-schema utilities for constructing validators."""


class SchemaKeys(object):
    """Commonly used type schema keys."""

    NAME = "@name"
    TYPE = "@type"
    INIT = "@init"
    NULLABLE = "@nullable"
    # collection keys
    MIN_ITEMS = "@min_items"
    MAX_ITEMS = "@max_items"
    ITEMS = "@items"
    # string keys
    MIN_LENGTH = "@min_length"
    MAX_LENGTH = "@max_length"
    # value bounds
    MINIMUM = "@minimum"
    EXCLUSIVE_MINIMUM = "@exclusive_minimum"
    MAXIMUM = "@maximum"
    EXCLUSIVE_MAXIMUM = "@exclusive_maximum"


class PropertiesKeys(object):
    """Common keys for value properties."""

    ID = "@id"
    HASH = "@hash"
    NAME = "@name"
    SCHEMA = "@schema"
    SCHEMAS = "@schemas"
    READONLY = "@readonly"


def UNSIGNED(default_value=None):
    """Unsigned integer value schema.

    Args:
        default_value: If not None `default` property is added.

    """
    result = {"type": "integer", "minimum": 0}
    if default_value is not None:
        result["default"] = default_value
    return result


def STRING_LIST(additional_properties={}):
    """Schema for array of strings (without `default`).

    Args:
        additional_properties: Added to result if specified.

    """
    result = {"type": "array", "items": {"type": "string"}}
    result.update(additional_properties)
    return result


def STRING_SCHEMA(additional_properties={}):
    """Schema for string value with length validation.

    Args:
        additional_properties: Added to result if specified.

    """
    result = {
        SchemaKeys.INIT: {"type": "string", "default": ""},
        SchemaKeys.MIN_LENGTH: UNSIGNED(),
        SchemaKeys.MAX_LENGTH: UNSIGNED(),
    }
    result.update(additional_properties)
    return result


def COLLECTION_SCHEMA(additional_properties={}):
    """Schema for collection with size validation.

    Args:
        additional_properties: Added to result if specified.

    """
    result = {SchemaKeys.MIN_ITEMS: UNSIGNED(), SchemaKeys.MAX_ITEMS: UNSIGNED()}
    result.update(additional_properties)
    return result


def BOUNDED_SCHEMA(type, additional_properties={}):
    """Schema for bounded scalar value.

    Args:
        type: Json schema type for minimum and maximum.
        additional_properties: Added to result if specified.

    """
    result = {
        "@minimum": {"type": type},
        "@exclusive_minimum": {"type": "boolean", "default": False},
        "@maximum": {"type": type},
        "@exclusive_maximum": {"type": "boolean", "default": False},
    }
    result.update(additional_properties)
    return result


"""
Value properties `definitions` parts of schema,
for links between schemas.
"""


def PROPERTIES_DEFINITIONS_SCHEMA_REF():
    """Link to definitions of 'one of types schemas'."""
    return {"$ref": "#/definitions/schema"}


def PROPERTIES_DEFINITIONS_SCHEMAS_REF():
    """Link to definitions of 'unique list of types schemas'."""
    return {"$ref": "#/definitions/schemas"}


def PROPERTIES_DEFINITIONS_TYPE_REF(type):
    """Link to `type` schema."""
    return {"$ref": f"#/definitions/types/{type.NAME}"}


"""
`definitions` section of properties schema, filled in constants module
by collecting schemas for all types.
"""
_PROPERTIES_DEFINITIONS = None


def _create_properties_definitions(types):
    return {
        "definitions": {
            "types": {type.NAME: type.schema() for type in types},
            "schema": {
                "oneOf": [PROPERTIES_DEFINITIONS_TYPE_REF(type) for type in types]
            },
            "schemas": {
                "type": "array",
                "uniqueItems": True,
                "items": {"$ref": "#/definitions/schema"},
            },
        }
    }


def PROPERTIES_SCHEMA(type):
    """Json-schema for validating of properties for `type`."""
    result = {
        "type": "object",
        "properties": {
            "@schema": PROPERTIES_DEFINITIONS_TYPE_REF(type),
            "@schemas": PROPERTIES_DEFINITIONS_SCHEMAS_REF(),
            "@name": {"type": "string", "description": "Value name"},
            "@id": {"type": "string", "description": "Unique value identifier"},
            "@hash": {"type": "integer", "description": "Value data hash"},
            "@readonly": {
                "type": "boolean",
                "default": False,
                "description": "Readonly flag for value editor",
            },
        },
        "required": ["@schema"],
        "description": "Value properties",
    }
    result.update(_PROPERTIES_DEFINITIONS)
    result["properties"].update(type.VALUE_PROPERTIES)
    return result
