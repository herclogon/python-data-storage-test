# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem subset type tests."""

from ..convert import convert
from ..value import Value
from .base_testcase import BaseTestCase


class TestGeneral(BaseTestCase):
    def setUp(self):
        super(TestGeneral, self).setUp()
        self.properties = {
            "@schema": {"@type": "Subset", "@values": ["foo", "bar", "zoo"]}
        }

    def test_read_write(self):
        self._read_write(["foo", "bar"], self.properties)

    def test_convert(self):
        value = Value(["foo", "bar"], self.properties)
        self._test_convert([(value, {"List": ["foo", "bar"], "NULL": None})])
        self.assertEqual(
            convert(self.properties["@schema"], value).native, value.native
        )

    def test_validation(self):
        Value(["foo", "bar"], self.properties).validate()
        with self.assertRaisesRegex(ValueError, "values"):
            Value(["foo", "bar", "asd"], self.properties).validate()
