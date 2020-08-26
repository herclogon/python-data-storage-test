# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem boolean type tests."""

import numpy

from ..convert import convert
from ..value import Value
from .base_testcase import BaseTestCase


class TestGeneral(BaseTestCase):
    CONVERT_RESULTS = [
        (
            True,
            {
                "Boolean": True,
                "NULL": None,
                "Integer": 1,
                "Real": 1.0,
                "String": "True",
                "List": [True],
                "Matrix": numpy.array([[1]], dtype=numpy.uint8),
            },
        ),
        (
            False,
            {
                "Boolean": False,
                "NULL": None,
                "Integer": 0,
                "Real": 0.0,
                "String": "False",
                "List": [False],
                "Matrix": numpy.array([[0]], dtype=numpy.uint8),
            },
        ),
    ]

    def test_read_write(self):
        self._read_write(True)
        self._read_write(False)

    def test_convert(self):
        self._test_convert(self.CONVERT_RESULTS)

    def test_convert_to_string(self):
        def _test(value, properties, result):
            self.assertEqual(
                convert(
                    {"@type": "String"}, Value(value, {"@schema": properties})
                ).native,
                result,
            )

        properties = {"@true_str": "trueString"}
        _test(True, properties, "trueString")
        _test(False, properties, "False")
        properties = {"@false_str": "falseString"}
        _test(True, properties, "True")
        _test(False, properties, "falseString")
        properties["@true_str"] = "trueString"
        _test(True, properties, "trueString")
        _test(False, properties, "falseString")
