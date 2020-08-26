# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem timestamp type tests."""

from datetime import datetime

from .. import schema
from ..types import timestamp
from ..value import Value
from .base_testcase import BaseTestCase


class TestGeneral(BaseTestCase):
    ISO_8601_FORMAT = "%Y-%m-%dT%H:%M:%S"

    def test_read_write(self):
        self._read_write(datetime.now().replace(microsecond=0))

        type_schema = {schema.SchemaKeys.TYPE: timestamp.Type.NAME}
        properties = {schema.PropertiesKeys.SCHEMA: type_schema}

        js_time = datetime.strftime(datetime.now(), self.ISO_8601_FORMAT) + "Z"
        for str_value in ["now", js_time]:
            self._read_write(Value(str_value, properties).native)

    def test_convert(self):
        now = datetime.now().replace(microsecond=0)
        str_now = datetime.strftime(now, self.ISO_8601_FORMAT)
        self._test_convert(
            [
                (
                    now,
                    {"List": [now], "String": str_now, "Timestamp": now, "NULL": None},
                ),
                (str_now, {"Timestamp": now}),
            ]
        )
