# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem dictionary type tests."""

from datetime import datetime

import pytest

from ..convert import convert
from ..file import File
from ..value import Value
from .base_testcase import BaseTestCase, TempPath


class TestGeneral(BaseTestCase):
    """Dictionary type tests."""

    DICTIONARY = {
        "asd": "foo",
        "43": 11.43,
        "bar": ["zoo", True, 2.1],
        "False": datetime.now().replace(microsecond=0),
    }

    def test_read_write(self):
        """Test serialization."""
        self._read_write(self.DICTIONARY)

    def test_convert(self):
        """Test value conversions."""
        self._test_convert(
            [(self.DICTIONARY, {"Dictionary": self.DICTIONARY, "NULL": None})]
        )

        dummy_schema = {
            "@type": "Structure",
            "@schema": {"asd": [{"@type": "String", "@name": "asd"}]},
        }
        self.assertEqual(convert(dummy_schema, self.DICTIONARY), self.DICTIONARY)

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
            Value({}, properties).validate()
        with self.assertRaisesRegex(ValueError, "higher"):
            Value({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, properties).validate()
        Value({"a": 1.0, "b": 2.0}, properties).validate()
        with self.assertRaisesRegex(TypeError, "type does not match"):
            Value({"a": 1.0, "b": "foo"}, properties).validate()
        properties["@schema"]["@items"].append({"@type": "String", "@name": "s"})
        value = Value({"a": 1.0, "b": "foo"}, properties)
        value.validate()

    def test_interface(self):
        """Test dictionary access methods."""
        v = Value(
            {
                "one": 1,
                "two": 2,
                "(1, 2, 3)": "tuple",
                "[1, 2, 3]": "list",
                '{1: "one", 2: "two"}': "dict",
            }
        )
        self.assertEqual(v.data["one"], 1)
        self.assertEqual(v.data["(1, 2, 3)"], "tuple")
        self.assertEqual(v.data[[1, 2, 3]], "list")
        self.assertEqual(v.data['{1: "one", 2: "two"}'], "dict")

        assert "two" in v.data
        v.data.pop("two")
        assert "two" not in v.data
        with pytest.raises(KeyError, match="two"):
            del v.data["two"]

        with TempPath() as path:
            with File(path.path, "w") as f:
                f.write(value=v)

                read_value = f.get()
                assert "[1, 2, 3]" in read_value.data
                del read_value.data["[1, 2, 3]"]
                assert "[1, 2, 3]" not in read_value.data
                with pytest.raises(KeyError, match="[1, 2, 3]"):
                    read_value.data.pop("[1, 2, 3]")

                read_value.data['{1: "one", 2: "two"}'] = "modified dict"

            with File(path.path, "r") as f:
                read_value = f.get()
                assert "two" not in read_value.data
                assert "[1, 2, 3]" not in read_value.data
                assert read_value.data['{1: "one", 2: "two"}'] == "modified dict"
