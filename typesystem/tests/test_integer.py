# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem integer type tests."""

import numpy

from ..value import Value
from .base_testcase import BaseTestCase


class TestGeneral(BaseTestCase):
    CONVERT_RESULTS = [
        (0, {"Boolean": False, "NULL": None}),
        (
            42,
            {
                "Boolean": True,
                "Integer": 42,
                "NULL": None,
                "List": [42],
                "Matrix": numpy.array([[42]], dtype=numpy.int64),
                "Real": 42.0,
                "String": "42",
            },
        ),
    ]

    def test_read_write(self):
        self._read_write(42)

    def test_convert(self):
        self._test_convert(self.CONVERT_RESULTS)

    def test_validation(self):
        properties = {"@schema": {"@minimum": 1, "@maximum": 3}}
        Value(1, properties).validate()
        Value(2, properties).validate()
        Value(3, properties).validate()
        with self.assertRaisesRegex(ValueError, "lower"):
            Value(0, properties).validate()
        with self.assertRaisesRegex(ValueError, "upper"):
            Value(4, properties).validate()
        properties["@schema"]["@exclusive_minimum"] = True
        with self.assertRaisesRegex(ValueError, "lower"):
            Value(1, properties).validate()
        properties["@schema"]["@exclusive_maximum"] = True
        with self.assertRaisesRegex(ValueError, "upper"):
            Value(3, properties).validate()
        Value(2, properties).validate()
