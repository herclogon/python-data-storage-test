# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem path type tests."""

from ..value import Value
from .base_testcase import BaseTestCase, TempPath


class TestGeneral(BaseTestCase):
    def setUp(self):
        super(TestGeneral, self).setUp()
        self.properties = {"@schema": {"@type": "Path"}}

    def test_read_write(self):
        self._read_write("foo.txt", self.properties)

    def test_convert(self):
        self._test_convert(
            [
                (
                    Value("foo", self.properties),
                    {"String": "foo", "List": ["foo"], "Path": "foo", "NULL": None},
                )
            ]
        )

    def test_validation(self):
        properties = {"@schema": {"@type": "Path", "@existing": True}}
        with TempPath() as path:
            with self.assertRaisesRegex(ValueError, "exist"):
                Value(path.path, properties).validate()
            with open(path.path, "w"):
                pass
            Value(path.path, properties).validate()
            properties["@schema"]["@file_type"] = "File"
            Value(path.path, properties).validate()
            properties["@schema"]["@file_type"] = "Any"
            Value(path.path, properties).validate()
            properties["@schema"]["@file_type"] = "Directory"
            with self.assertRaisesRegex(ValueError, "type"):
                Value(path.path, properties).validate()
