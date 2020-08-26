# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem enumeration type tests."""

from ..convert import convert
from ..value import Value
from .base_testcase import BaseTestCase


class TestGeneral(BaseTestCase):
    def setUp(self):
        super(TestGeneral, self).setUp()
        self.properties = {
            "@schema": {"@type": "Enumeration", "@values": ["foo", "bar", "zoo"]}
        }

    def test_read_write(self):
        self._read_write("foo", self.properties)

    def test_convert(self):
        value = Value("foo", self.properties)
        self._test_convert([(value, {"String": "foo", "List": ["foo"], "NULL": None})])
        self.assertEqual(
            convert(self.properties["@schema"], value).native, value.native
        )

    def test_validation(self):
        Value("foo", self.properties).validate()
        with self.assertRaisesRegex(ValueError, "enumeration"):
            Value("asd", self.properties).validate()
