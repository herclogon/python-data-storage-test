# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem list type tests."""

from datetime import datetime

import numpy

from .. import file
from ..convert import convert
from ..value import Value
from .base_testcase import BaseTestCase, TempPath


class TestGeneral(BaseTestCase):
    """List type tests."""

    LIST = [
        42,
        False,
        "foo",
        datetime.now().replace(microsecond=0),
        {"foo": 12, "3": False},
        slice(3, 2, 1),
    ]

    def test_read_write(self):
        """Test serialization."""
        self._read_write(self.LIST)

    def test_modifications(self):
        """Test list modification."""
        value = Value(self.LIST)
        value.data.pop(2)
        self.assertEqual(value.data[2], self.LIST[3])
        value.data[2] = datetime(2019, 12, 31)
        self.assertEqual(value.data[2], datetime(2019, 12, 31))

    def test_modifications_file(self):
        """Test modification of a list bound to a file."""
        value = Value(self.LIST)
        with TempPath() as path:
            with file.File(path.path, "w") as f:
                f.write(value=value)
                id = f.ids()[0]
                # replace entire list
                f[id] = range(5)
                # change an item in the new list - won't affect the file!
                # because Value at f[id] is new and isn't bound with the file yet
                f.get(id).data[3] = 128
                self.assertEqual(f[id][3], 128)

            result = f.read()
            self.assertEqual(result, [0, 1, 2, 3, 4])

            with file.File(path.path, "w") as f:
                f.get(id).data.pop(2)
            result = f.read()
            self.assertEqual(result, [0, 1, 3, 4])

    def test_convert(self):
        """Test value conversions."""
        self._test_convert([(self.LIST, {"List": self.LIST, "NULL": None})])
        self._test_convert(
            [
                (
                    [True],
                    {"Boolean": True, "Integer": 1, "Real": 1.0, "String": "True"},
                ),
                (
                    [False],
                    {"Boolean": False, "Integer": 0, "Real": 0.0, "String": "False"},
                ),
            ]
        )
        self._test_convert([(["foo"], {"String": "foo", "Path": "foo"})])
        self._test_convert(
            [([42], {"Boolean": True, "Integer": 42, "Real": 42, "String": "42"})]
        )
        now = datetime.now().replace(microsecond=0)
        self._test_convert([([3.14], {"Real": 3.14}), ([now], {"Timestamp": now})])
        self._test_convert(
            [([1, 2, 3], {"Matrix": numpy.array([[1, 2, 3]], dtype=numpy.int64)})]
        )
        self._test_convert(
            [
                (
                    [1.1, 2.2, 3.3],
                    {"Matrix": numpy.array([[1.1, 2.2, 3.3]], dtype=numpy.float64)},
                )
            ]
        )
        self._test_convert(
            [
                (
                    [1, 2, 3.3],
                    {"Matrix": numpy.array([[1.0, 2.0, 3.3]], dtype=numpy.float64)},
                )
            ]
        )

        self.assertEqual(
            convert({"@type": "Enumeration", "@values": ["foo"]}, ["foo"]), "foo"
        )

    def test_convert_error(self):
        """Test value conversions which must produce errors."""

        self._test_convert_error(
            [
                (
                    [1.1, 2.2, 3.3],
                    {
                        "Dictionary": "no way to convert List to Dictionary",
                        "Integer": "list size must be exactly 1",
                    },
                )
            ]
        )

    def test_validation(self):
        """Test value validation."""
        properties = {
            "@schema": {
                "@min_items": 1,
                "@max_items": 3,
                "@items": [{"@type": "Real", "@name": "r"}],
            }
        }
        with self.assertRaisesRegex(ValueError, "lower"):
            Value([], properties).validate()
        with self.assertRaisesRegex(ValueError, "higher"):
            Value([1.0, 2.0, 3.0, 4.0], properties).validate()
        Value([1.0], properties).validate()
        Value([1.0, 2.0], properties).validate()
        Value([1.0, 2.0, 3.0], properties).validate()
        properties["@schema"]["@unique_items"] = True
        Value([1.0, 2.0, 3.0], properties).validate()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            Value([1.0, 1.0], properties).validate()
        with self.assertRaisesRegex(TypeError, "type does not match"):
            Value([1.0, "foo"], properties).validate()
        properties["@schema"]["@items"].append({"@type": "String", "@name": "s"})
        Value([1.0, "foo"], properties).validate()
