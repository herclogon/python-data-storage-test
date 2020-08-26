# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem string type tests."""

from ..convert import convert
from ..value import Value
from .base_testcase import BaseTestCase


class TestGeneral(BaseTestCase):
    def test_read_write(self):
        self._read_write("foobar")
        self._read_write("авпвапвкп")

    def test_convert(self):
        self._test_convert(
            [
                (
                    "foo",
                    {"String": "foo", "List": ["foo"], "NULL": None, "Path": "foo"},
                ),
                ("42", {"Integer": 42, "Real": 42.0}),
                ("3.14", {"Real": 3.14}),
            ]
        )
        for s in ["True", "TRUE", "true", "1", "Yes", "YES", "yes"]:
            self._test_convert([(s, {"Boolean": True})])
        for s in ["False", "FALSE", "false", "0", "No", "NO", "no"]:
            self._test_convert([(s, {"Boolean": False})])

        self.assertEqual(
            convert({"@type": "Enumeration", "@values": ["foo"]}, "foo"), "foo"
        )

        self.assertEqual(convert({"@type": "Boolean", "@true_str": "asd"}, "asd"), True)
        self.assertEqual(
            convert({"@type": "Boolean", "@true_str": "asd"}, "False"), False
        )
        self.assertEqual(
            convert({"@type": "Boolean", "@false_str": "asd"}, "asd"), False
        )
        self.assertEqual(
            convert({"@type": "Boolean", "@false_str": "asd"}, "True"), True
        )

        with self.assertRaisesRegex(ValueError, "not convertible"):
            convert({"@type": "Boolean", "@false_str": "asd"}, "False")
        with self.assertRaisesRegex(ValueError, "not convertible"):
            convert({"@type": "Boolean", "@true_str": "asd"}, "True")

    def test_validation(self):
        properties = {"@schema": {"@min_length": 3, "@max_length": 7}}
        Value("asdfs", properties).validate()
        Value("asd", properties).validate()
        Value("asdfghj", properties).validate()
        with self.assertRaisesRegex(ValueError, "lower"):
            Value("a", properties).validate()
        with self.assertRaisesRegex(ValueError, "greater"):
            Value("asdfghjk", properties).validate()
        properties["@schema"]["@pattern"] = "^\\d+$"
        Value("123", properties).validate()
        with self.assertRaisesRegex(ValueError, "pattern"):
            Value("123ad", properties).validate()
