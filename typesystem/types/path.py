# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem path type."""

import os

from . import string
from .. import schema
from .common import nativeTypeWithAnotherAsStorage


class Type(nativeTypeWithAnotherAsStorage(string.Type)):
    """Path type implementation."""

    NAME = "Path"

    SCHEMA = {
        schema.SchemaKeys.INIT: {"type": "string", "default": ""},
        "@file_type": {
            "type": "string",
            "enum": ["Any", "File", "Directory"],
            "default": "Any",
        },
        "@existing": {"type": "boolean", "default": False},
        "@formats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "minLength": 1},
                    "value": {"type": "string", "minLength": 1},
                },
                "required": ["label", "value"],
            },
            "default": [],
        },
    }

    VALUE_PROPERTIES = {
        "@format": {"type": "integer", "minimum": 0},
        "@encoding": {"type": "string", "minLength": 1},
        "@ending": {"type": "string", "enum": ["CR", "LF", "CRLF"]},
    }

    @classmethod
    def validate(cls, value, schema, errors_list):
        super(Type, cls).validate(value, schema, errors_list)
        if schema.get(
            "@existing", cls.SCHEMA["@existing"]["default"]
        ) and not os.path.exists(value):
            errors_list.append(ValueError("path does not exist '%s'" % value))
        file_type = schema.get("@file_type", cls.SCHEMA["@file_type"]["default"])
        if (
            file_type != "Any"
            and os.path.exists(value)
            and os.path.isfile(value) != (file_type == "File")
        ):
            errors_list.append(
                ValueError(
                    "value type does not match required: %s" % (file_type.lower())
                )
            )
