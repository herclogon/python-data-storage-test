# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem slice type tests."""

from ..value import Value
from .base_testcase import BaseTestCase


class TestGeneral(BaseTestCase):
    def test_read_write(self):
        self._read_write(slice(1, 10, 3))
        self._read_write(
            slice(1, 10),
            compare=lambda value, result: self.assertEqual(result, slice(1, 10, 1)),
        )
        self._read_write(
            slice(10),
            compare=lambda value, result: self.assertEqual(result, slice(0, 10, 1)),
        )

    def test_convert(self):
        data = slice(1, 10, 3)
        self._test_convert([(data, {"List": [data], "Slice": data, "NULL": None})])

    def test_validation(self):
        properties = {"@schema": {"@minimum": 1, "@maximum": 7}}
        Value(slice(1, 8, 2), properties).validate()
        Value(slice(1, 8, 3), properties).validate()
        Value(slice(1, 8, 4), properties).validate()
        with self.assertRaisesRegex(ValueError, "lower"):
            Value(slice(0, 8, 4), properties).validate()
        with self.assertRaisesRegex(ValueError, "upper"):
            Value(slice(1, 9, 4), properties).validate()
        properties["@schema"]["@exclusive_minimum"] = True
        with self.assertRaisesRegex(ValueError, "lower"):
            Value(slice(1, 8, 4), properties).validate()
        properties["@schema"]["@exclusive_maximum"] = True
        with self.assertRaisesRegex(ValueError, "upper"):
            Value(slice(2, 8, 1), properties).validate()
        Value(slice(6, 2, -1), properties).validate()
        Value(slice(6, 1, -2), properties).validate()
