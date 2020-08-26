# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem structure type tests."""

import copy
from datetime import datetime

import pytest

from ..convert import convert
from ..file import File
from ..value import Value
from .base_testcase import BaseTestCase, TempPath


class TestGeneral(BaseTestCase):
    """Structure type tests."""

    STRUCTURE = {
        "data": 3.14,
        "time": datetime.now().replace(microsecond=0),
        "count": 42,
    }

    def setUp(self):
        super(TestGeneral, self).setUp()
        self.schema = {
            "data": [{"@type": "Real", "@name": "result", "@init": float("nan")}],
            "time": [
                {"@type": "Timestamp", "@name": "moment of glory", "@init": "now"}
            ],
            "count": [{"@type": "Integer", "@name": "attempt number", "@init": -1}],
        }
        self.properties = {
            "@schema": {
                "@type": "Structure",
                "@items": self.schema,
                "@allow_missing_properties": False,
                "@allow_additional_properties": False,
            }
        }

    def test_read_write(self):
        """Test serialization."""
        self._read_write(self.STRUCTURE, self.properties)

    def test_convert(self):
        """Test value conversions."""
        value = Value(self.STRUCTURE, self.properties)
        self._test_convert([(value, {"Dictionary": self.STRUCTURE, "NULL": None})])
        self.assertEqual(
            convert(self.properties["@schema"], value).native, value.native
        )

    def test_validation(self):
        """Test value validation."""
        properties = copy.deepcopy(self.properties)
        data = copy.deepcopy(self.STRUCTURE)
        Value(data, properties).validate()
        data.pop("data")
        with self.assertRaisesRegex(ValueError, "missing"):
            Value(data, properties).validate()
        properties["@schema"]["@allow_missing_properties"] = True
        Value(data, properties).validate()
        data["foo"] = "bar"
        with self.assertRaisesRegex(ValueError, "additional"):
            Value(data, properties).validate()
        properties["@schema"]["@allow_additional_properties"] = True
        Value(data, properties).validate()
        data["count"] = "barz"
        with self.assertRaisesRegex(TypeError, "type does not match"):
            Value(data, properties).validate()
        properties["@schema"]["@items"]["count"].append(
            {"@type": "String", "@name": "wrong type"}
        )
        Value(data, properties).validate()

    def test_interface(self):
        v = Value(self.STRUCTURE, self.properties)

        assert "data" in v.data
        v.data.pop("data")
        assert "data" not in v.data
        with pytest.raises(KeyError, match="data"):
            del v.data["data"]

        with TempPath() as path:
            with File(path.path, "w") as f:
                f.write(value=v)

                read_value = f.get()
                assert "time" in read_value.data
                read_value.data.pop("time")
                assert "time" not in read_value.data
                with pytest.raises(KeyError, match="time"):
                    del read_value.data["time"]

                read_value.data["count"] = 12345

            with File(path.path, "r") as f:
                read_value = f.get()
                assert "data" not in read_value.data
                assert "time" not in read_value.data
                assert read_value.data["count"] == 12345
