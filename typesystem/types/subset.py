# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem subset type."""

from . import list
from .. import schema
from .common import nativeTypeWithAnotherAsStorage


class Type(nativeTypeWithAnotherAsStorage(list.Type)):
    """Subset type implementation."""

    NAME = "Subset"

    _SCHEMA = {
        schema.SchemaKeys.INIT: schema.STRING_LIST(
            {"default": [], "uniqueItems": True}
        ),
        "@values": schema.STRING_LIST({"uniqueItems": True}),
    }
    SCHEMA = schema.COLLECTION_SCHEMA(_SCHEMA)
    SCHEMA_REQUIRED = ["@values"]

    @classmethod
    def validate(cls, value, schema, errors_list):
        super(Type, cls).validate(value, schema, errors_list)
        cls._validate_collection_size(value, schema, errors_list)
        if "@values" in schema:
            missing = ["{}".format(v) for v in value if v not in set(schema["@values"])]
            if len(missing):
                errors_list.append(
                    ValueError(
                        "values are not in schema: ['%s']" % "', '".join(missing)
                    )
                )
