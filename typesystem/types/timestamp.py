# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem timestamp type."""

from datetime import datetime

import numpy
import pytz

from .. import schema
from .common import SimpleHDFType


class Type(SimpleHDFType):
    """Timestamp type implementation."""

    NAME = "Timestamp"
    NATIVE_TYPE = datetime
    _STORAGE_DTYPE = (numpy.float64,)

    SCHEMA = {schema.SchemaKeys.INIT: {"type": "string", "default": "now"}}

    VALUE_PROPERTIES = {
        "@timezone": {"type": "string", "pattern": "^([+,-])?\\d\\d:\\d\\d$"}
    }

    @classmethod
    def from_native(cls, value, properties):
        if isinstance(value, str):
            if value == "now":
                return cls.NATIVE_TYPE.utcnow()
            # `datetime` does not support timezone military suffixes.
            # But JavaScript always returns ISO string with trailing Z.
            if value and value[-1] == "Z":
                value = value[:-1]
            value = cls.NATIVE_TYPE.fromisoformat(value)
        # Get timezone unaware UTC timestamp.
        return value.replace(tzinfo=pytz.utc).replace(tzinfo=None)

    @classmethod
    def raw_to_native(cls, value, properties):
        return cls.NATIVE_TYPE.utcfromtimestamp(value[0])

    @classmethod
    def native_to_raw(cls, value, properties):
        utc = value.replace(tzinfo=pytz.utc).timestamp()
        return [cls._STORAGE_DTYPE[0](utc)]

    @classmethod
    def from_hdf_column(cls, data, column_properties):
        return numpy.array(
            [cls.raw_to_native(item, column_properties) for item in data],
            dtype=numpy.object,
        )
