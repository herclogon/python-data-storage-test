# Copyright (C) DATADVANCE, 2010-2020


"""pSeven typesystem real type tests."""

import numpy

from ..value import Value
from .base_testcase import BaseTestCase


class TestGeneral(BaseTestCase):
    def test_read_write(self):
        self._read_write(3.14)
        self._read_write(
            float("NaN"),
            compare=lambda value, result: self.assertTrue(numpy.isnan(result)),
        )
        self._read_write(
            float("Inf"),
            compare=lambda value, result: self.assertTrue(
                numpy.isinf(result) and result > 0
            ),
        )
        self._read_write(
            float("-Inf"),
            compare=lambda value, result: self.assertTrue(
                numpy.isinf(result) and result < 0
            ),
        )

    def test_convert(self):
        self._test_convert(
            [
                (
                    3.14,
                    {
                        "Boolean": True,
                        "Integer": 3,
                        "List": [3.14],
                        "Matrix": numpy.array([[3.14]]),
                        "String": "3.14",
                        "Real": 3.14,
                        "NULL": None,
                    },
                ),
                (0.0, {"Boolean": False, "Integer": 0, "Real": 0.0, "NULL": None}),
            ]
        )

    def test_validation(self):
        properties = {"@schema": {"@minimum": 1.0, "@maximum": 3.0}}
        Value(1.0, properties).validate()
        Value(2.0, properties).validate()
        Value(3.0, properties).validate()
        with self.assertRaisesRegex(ValueError, "lower"):
            Value(0.0, properties).validate()
        with self.assertRaisesRegex(ValueError, "upper"):
            Value(4.0, properties).validate()
        properties["@schema"]["@exclusive_minimum"] = True
        with self.assertRaisesRegex(ValueError, "lower"):
            Value(1.0, properties).validate()
        properties["@schema"]["@exclusive_maximum"] = True
        with self.assertRaisesRegex(ValueError, "upper"):
            Value(3.0, properties).validate()
        Value(2.0, properties).validate()

        Value(float("NaN"), properties).validate()
        properties["@schema"]["@nanable"] = False
        with self.assertRaisesRegex(ValueError, "NaN"):
            Value(float("NaN"), properties).validate()
