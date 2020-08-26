# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem enumeration type."""

from . import string
from .. import schema
from .common import nativeTypeWithAnotherAsStorage


class Type(nativeTypeWithAnotherAsStorage(string.Type)):
    """Enumeration type implementation.

    All functions expect instance of Dictionary object as value-object.
    """

    NAME = "Enumeration"
    SCHEMA = {
        schema.SchemaKeys.INIT: {"type": "string", "default": ""},
        "@values": schema.STRING_LIST({"minItems": 1}),
    }
    SCHEMA_REQUIRED = ["@values"]

    @classmethod
    def validate(cls, value, value_schema, errors_list):
        super(Type, cls).validate(value, value_schema, errors_list)
        if value not in value_schema["@values"]:
            errors_list.append(ValueError("value not in enumeration list"))
