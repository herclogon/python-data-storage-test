# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem table type tests."""

from datetime import datetime
import json
import random
import time

import h5py
import numpy
import pytest

from .. import file
from ..convert import convert
from ..types import table
from ..value import Value
from .base_testcase import BaseTestCase, TempPath


class TestGeneral(BaseTestCase):
    """Table type tests."""

    @staticmethod
    def _random_str(size, start, end):
        return "".join(
            map(
                chr,
                [
                    random.randint(ord(start), ord(end))
                    for _ in range(random.randint(0, size))
                ],
            )
        )

    def _random_ascii(self, size):
        return self._random_str(size, "a", "z")

    def _random_unicode(self, size):
        return self._random_str(size, "а", "я")

    def test_read_write_simple(self):
        """Test serialization basic usage."""
        dtype = [("Col 1", numpy.float64), ("Col 2", numpy.float64)]
        self._read_write(numpy.random.rand(3).astype(dtype))
        self._read_write(numpy.array([], dtype=dtype))

    def test_read_write_binary_column(self):
        """Test serialization with binary data."""
        self._read_write(
            numpy.array(
                [(3.14, "hello"), (2.71, "world")],
                dtype=[("Col 1", numpy.float64), ("Col 2", "S16")],
            )
        )

    def test_read_write_string_column(self):
        """Test serialization with string data."""
        self._read_write(
            numpy.array(
                [(3.14, "hello"), (2.71, "world")],
                dtype=[("Col 1", numpy.float64), ("Col 2", "O")],
            )
        )

    def test_convert(self):
        """Test value conversions."""
        data = numpy.array(
            [(3.14, 3.15), (2.71, 2.72)],
            dtype=[("Col 1", numpy.float64), ("Col 2", numpy.float64)],
        )
        self._test_convert(
            [
                (
                    data,
                    {
                        "Matrix": numpy.array([list(r) for r in data]),
                        "Table": data,
                        "NULL": None,
                    },
                )
            ]
        )
        self._test_convert(
            [
                (  # one row Table to List
                    numpy.array(
                        [(3.14, 3.15)],
                        dtype=[("Col 1", numpy.float64), ("Col 2", numpy.float64)],
                    ),
                    {"List": [3.14, 3.15]},
                ),
                (  # one column Table to List
                    numpy.array([(3.14,), (2.71,)], dtype=[("Col 1", numpy.float64)]),
                    {"List": [3.14, 2.71]},
                ),
            ]
        )
        self._test_convert(
            [
                (
                    numpy.array([(42,)], dtype=[("Col 1", numpy.int64)]),
                    {"Boolean": True, "Integer": 42, "Real": 42.0, "String": "42"},
                )
            ]
        )
        self._test_convert(
            [
                (
                    numpy.array([(0.0,)], dtype=[("Col 1", numpy.float64)]),
                    {"Boolean": False, "Integer": 0.0, "Real": 0.0, "String": "0.0"},
                )
            ]
        )
        self._test_convert(
            [
                (
                    numpy.array([(True,)], dtype=[("Col 1", numpy.bool_)]),
                    {"Boolean": True, "Integer": 1, "Real": 1.0, "String": "True"},
                )
            ]
        )
        self._test_convert(
            [
                (
                    numpy.array([(False,)], dtype=[("Col 1", numpy.bool_)]),
                    {"Boolean": False, "Integer": 0, "Real": 0.0, "String": "False"},
                )
            ]
        )
        self._test_convert(
            [
                (
                    numpy.array([("foo",)], dtype=[("Col 1", "U10")]),
                    {"String": "foo", "Path": "foo"},
                )
            ]
        )
        self._test_convert(
            [
                (
                    numpy.array([("42",)], dtype=[("Col 1", "U10")]),
                    {"Integer": 42, "Real": 42.0, "String": "42"},
                )
            ]
        )

        data = numpy.array(
            [(3.14, "zoo")], dtype=[("Col 1", numpy.float64), ("Col 2", "U11")]
        )
        as_dict = {"Col 1": 3.14, "Col 2": "zoo"}
        self._test_convert([(data, {"Dictionary": as_dict})])
        self.assertEqual(
            as_dict,
            convert(
                {
                    "@type": "Structure",
                    "@schema": {"foo": [{"@type": "String", "@name": "foo"}]},
                },
                data,
            ),
        )

        self.assertEqual(
            "foo",
            convert(
                {"@type": "Enumeration", "@values": ["foo"]},
                numpy.array([("foo",)], dtype=[("Col 1", "U10")]),
            ),
        )

    def test_convert_error(self):
        """Test value conversions which must produce errors."""

        data = numpy.array(
            [(3.14, "3.15"), (2.71, "2.72")],
            dtype=[("Col 1", numpy.float64), ("Col 2", numpy.str_)],
        )

        self._test_convert_error(
            [
                (
                    data,
                    {
                        "List": "table must have one row or one column",
                        "Dictionary": "table must have exactly one row",
                        "Matrix": "impossible to convert table to matrix, wrong column types",
                        "Integer": "table size must be exactly 1x1",
                    },
                )
            ]
        )

        self._test_convert_error([(None, {"Table": "no way to convert NULL to Table"})])

    def test_row_append_simple(self):
        """Test append rows to the table."""
        data = numpy.array(
            [(3.14, 3.15), (2.71, 2.72)],
            dtype=[("Col 1", numpy.float64), ("Col 2", numpy.float64)],
        )
        with TempPath() as path:
            f = file.File(path.path, "w")
            f.write(value=data)
            new_data = numpy.random.rand(3, 2)
            with f:
                table = f.get()
                table.data.append(new_data[0])
                table.data.append_rows(new_data[1:])
            self.assertEqual(
                f.read(),
                numpy.append(
                    data, numpy.array([tuple(r) for r in new_data], dtype=data.dtype)
                ),
            )

    def test_row_append_to_empty(self):
        """Test append rows to an empty table."""
        data = numpy.array(
            [],
            dtype=[
                ("Col 1", numpy.float64),
                ("Col 2", numpy.float64),
                ("Col 3", "S16"),
            ],
        )
        with TempPath() as path:
            f = file.File(path.path, "w")
            f.write(value=data)
            new_data = [
                [3.14, 2.71, "foobar"],
                [2.34, 7.15, "bazzfuzzzzzz"],
                [4.0, 42.0, "zoo"],
            ]
            with f:
                table = f.get()
                table.data.append(new_data[0])
                table.data.append_rows(new_data[1:])
            self.assertEqual(
                f.read(),
                numpy.append(
                    data, numpy.array([tuple(r) for r in new_data], dtype=data.dtype)
                ),
            )

    def test_append_complex(self):
        """Test append rows with different types."""
        dtype = [
            ("Col 1", numpy.float64),
            ("Col 2", numpy.bool),
            ("Col 3", numpy.object),
            ("Col 4", "S8"),
            ("Col 5", "U8"),
        ]
        data = [(3.14, True, "asdas", "asds", "sssss")]
        with TempPath() as path:
            f = file.File(path.path, "w")
            f.write(value=numpy.array(data, dtype=dtype))
            with f:
                table = f.get()
                for _ in range(19):
                    row = [
                        random.random(),
                        bool(random.randint(0, 1)),
                        self._random_unicode(128),
                        self._random_ascii(8),
                        self._random_unicode(8),
                    ]
                    table.data.append(row)
                    data.append(tuple(row))

                dtype_after_read = [
                    ("Col 1", numpy.float64),
                    ("Col 2", numpy.bool),
                    ("Col 3", numpy.object),
                    ("Col 4", numpy.object),
                    ("Col 5", numpy.object),
                ]
                data = numpy.array(data, dtype=dtype_after_read)
                data["Col 4"] = [bytes(item, "utf-8") for item in data["Col 4"]]

                self.assertEqual(f.get().native, data)
            self.assertEqual(f.read(), data)

    def test_mask_api(self):
        """Test masked api (None values)."""
        dtype = [("Col 1", numpy.float64), ("Col 2", "S8")]

        def mask(value):
            return numpy.array(
                [tuple(r) for r in value],
                dtype=[("Col 1", numpy.bool_), ("Col 2", numpy.bool_)],
            )

        with TempPath() as path:
            f = file.File(path.path, "w")
            f.write(value=numpy.array([], dtype=dtype))
            with f:
                table = f.get().data
                table.append([3.14, "xoo"])
                self.assertEqual(table.mask, mask([[False, False]]))
                table.append([3.14, "xoo"], [False, True])
                self.assertEqual(table.mask, mask([[False, False], [False, True]]))
                table.append(
                    numpy.ma.masked_array(
                        [(2.71, "zoo")], mask=[(True, False)], dtype=dtype
                    )
                )
                self.assertEqual(
                    table.mask, mask([[False, False], [False, True], [True, False]])
                )

                table[1] = [3.14, "foo"]
                self.assertEqual(
                    table.mask, mask([[False, False], [False, False], [True, False]])
                )

                table.append_rows([[3.14, "foo"]])
                self.assertEqual(
                    table.mask,
                    mask(
                        [[False, False], [False, False], [True, False], [False, False]]
                    ),
                )
                table.append_rows([[3.14, "foo"]], [[True, False]])
                self.assertEqual(
                    table.mask,
                    mask(
                        [
                            [False, False],
                            [False, False],
                            [True, False],
                            [False, False],
                            [True, False],
                        ]
                    ),
                )
                table.append_rows(
                    numpy.ma.masked_array(
                        [(2.71, "zoo")], mask=[(True, False)], dtype=dtype
                    )
                )
                self.assertEqual(
                    table.mask,
                    mask(
                        [
                            [False, False],
                            [False, False],
                            [True, False],
                            [False, False],
                            [True, False],
                            [True, False],
                        ]
                    ),
                )

                table.cell_set(2, 0, 1.1)
                self.assertEqual(
                    table.mask,
                    mask(
                        [
                            [False, False],
                            [False, False],
                            [False, False],
                            [False, False],
                            [True, False],
                            [True, False],
                        ]
                    ),
                )

    def test_cell_access(self):
        """Test access to single cell."""
        dtype = [
            ("Col 1", numpy.float64),
            ("Col 2", numpy.bool),
            ("Col 3", numpy.object),
        ]
        data = [
            (3.14, True, self._random_unicode(128)),
            (2.71, False, self._random_unicode(128)),
        ]
        with TempPath() as path:
            f = file.File(path.path, "w")
            f.write(value=numpy.array(data, dtype=dtype))
            data = data[-1::-1]
            with f:
                table = f.get().data
                for row_idx, row in enumerate(data):
                    for col_idx, cell in enumerate(row):
                        table.cell_set(row_idx, col_idx, cell)
                        self.assertEqual(table[row_idx, col_idx], cell)
                        self.assertEqual(table[row_idx, dtype[col_idx][0]], cell)
                self.assertEqual(table[-1], numpy.array([data[-1]], dtype=dtype))
                self.assertEqual(table[-1, -2], data[-1][-2])
                self.assertEqual(f.read(), numpy.array(data, dtype=dtype))
                data = data[-1::-1]
                for row_idx, row in enumerate(data):
                    for col_idx, cell in enumerate(row):
                        table[row_idx, col_idx] = cell
                        table[row_idx, dtype[col_idx][0]] = cell
                self.assertEqual(f.read(), numpy.array(data, dtype=dtype))
            self.assertEqual(f.read(), numpy.array(data, dtype=dtype))

    def test_string_overflow(self):
        """Test string columns size overflow."""
        dtype = [("Col 1", "S2"), ("Col 2", "U1")]
        read_dtype = [("Col 1", "O"), ("Col 2", "O")]
        # @TODO do we have to keep the original dtype?
        with TempPath() as path:
            f = file.File(path.path, "w")
            f.write(value=numpy.array([("aaa", "ыы")], dtype=dtype))
            with f:
                self.assertEqual(f.read(), numpy.array([(b"aa", "ы")], dtype=dtype))
                f.get().data[0] = ["sss", "яяфф"]
                self.assertEqual(
                    f.read(), numpy.array([(b"sss", "яяфф")], dtype=read_dtype)
                )
            self.assertEqual(
                f.read(), numpy.array([(b"sss", "яяфф")], dtype=read_dtype)
            )

    def test_column_properties(self):
        """Test properties of individual column."""
        properties = {
            "@columns": {
                "0": {"@schema": {"@type": "Enumeration", "@values": ["a", "v"]}}
            }
        }
        dtype = [("Enum", "U10")]
        with TempPath() as path:
            f = file.File(path.path, "w")
            f.write(
                value=numpy.array([("aaa",), ("vvv",)], dtype=dtype),
                properties=properties,
            )
            with file.File(path.path, "w") as f:
                self.assertEqual(
                    f.get().data.column_properties("Enum")["@schema"]["@type"],
                    "Enumeration",
                )
                self.assertEqual(
                    f.get().data.column_properties(0)["@schema"]["@type"], "Enumeration"
                )
                f.get().data.append(numpy.array([("zzz",)], dtype=dtype))
        with TempPath() as path:
            f = file.File(path.path, "w")
            f.write(value=numpy.array([], dtype=dtype), properties=properties)
            with file.File(path.path, "w") as f:
                self.assertEqual(
                    f.get().data.column_properties("Enum")["@schema"]["@type"],
                    "Enumeration",
                )
                f.get().data.append(["zzz"], mask=[True])

        properties["@columns"]["0"]["@name"] = "Enumeration data"
        v = Value(numpy.array([], dtype=dtype), properties)
        self.assertEqual(v.data.dtype.names[0], "Enumeration data")
        self.assertEqual(v.native.dtype.names[0], "Enumeration data")

    def test_rows_selection(self):
        """Test data retrieving in windowed mode."""
        dtype = [("1", numpy.float64), ("2", numpy.object), ("3", numpy.int64)]
        data = numpy.array(
            [
                (random.random(), self._random_unicode(13), random.randint(0, 9))
                for _ in range(7)
            ],
            dtype=dtype,
        )
        with TempPath() as path:
            file.File(path.path, "w").write(value=data)
            with file.File(path.path, "r") as f:
                table = f.get().data
                for c in data.dtype.names:
                    self.assertEqual(data[c], table.rows(columns=[c])[c])
                rows = table.rows(columns=["1", "3"])
                for c in rows.dtype.names:
                    self.assertEqual(data[c], rows[c])
                rows = table.rows(0, 3, columns=["1", "2"])
                for c in rows.dtype.names:
                    self.assertEqual(data[c][0:3], rows[c])
                rows = table.rows(4, None, columns=["3", "2"])
                self.assertEqual(rows.dtype.names, ("3", "2"))
                for c in rows.dtype.names:
                    self.assertEqual(data[c][4 : len(data[c])], rows[c])

                rows = table.matrix(["1"])
                self.assertEqual(rows.shape, (len(data), 1))
                self.assertEqual(rows[:, 0], data["1"])
                rows = table.matrix(["3"])
                self.assertEqual(rows.shape, (len(data), 1))
                self.assertEqual(rows[:, 0], data["3"])
                rows = table.matrix(["3", "1"])
                self.assertEqual(rows.shape, (len(data), 2))
                self.assertEqual(rows[:, 0], data["3"])
                self.assertEqual(rows[:, 1], data["1"])

    def test_large_table(self):
        """Test performance of a large table.

        @cmake_labels long
        """
        ncols = 100
        nrows = 100000
        data = numpy.array(
            [tuple(row) for row in numpy.random.uniform(0, 1, (nrows, ncols))],
            dtype=[("Col %d" % idx, numpy.float64) for idx in range(ncols)],
        )
        with TempPath() as path:
            tm = time.time()
            file.File(path.path, "w").write(value=data)
            print("Large table write finished in %f" % (time.time() - tm))
            tm = time.time()
            read = file.File(path.path, "r").read()
            print("Large table read finished in %f" % (time.time() - tm))
            tm = time.time()
            self.assertEqual(read, data)
            print("Large table compare finished in %f" % (time.time() - tm))

    def test_validation(self):
        """Test value validation."""
        dtype = [("1", numpy.float64), ("2", numpy.object), ("3", "U12")]
        data = numpy.array(
            [
                (random.random(), self._random_unicode(13), self._random_unicode(9))
                for _ in range(3)
            ],
            dtype=dtype,
        )
        Value(data).validate()

        properties = {"@schema": {"@min_cols": 4}}
        with self.assertRaisesRegex(ValueError, "columns.+less"):
            Value(data, properties).validate()
        properties = {"@schema": {"@min_cols": 1, "@max_cols": 1}}
        with self.assertRaisesRegex(ValueError, "columns.+more"):
            Value(data, properties).validate()
        properties = {"@schema": {"@min_cols": 1, "@max_cols": 4, "@min_rows": 12}}
        with self.assertRaisesRegex(ValueError, "rows.+less"):
            Value(data, properties).validate()
        properties = {
            "@schema": {"@min_cols": 1, "@max_cols": 4, "@min_rows": 1, "@max_rows": 1}
        }
        with self.assertRaisesRegex(ValueError, "rows.+more"):
            Value(data, properties).validate()
        properties["@schema"]["@max_rows"] = 12
        Value(data, properties).validate()

        properties["@schema"]["@columns"] = {"0": [{"@type": "Real", "@name": "r"}]}
        Value(data, properties).validate()
        properties["@schema"]["@columns"]["0"][0]["@minimum"] = 3.0
        with self.assertRaisesRegex(ValueError, "lower"):
            Value(data, properties).validate()
        properties["@schema"]["@columns"]["0"][0]["@minimum"] = 0.0
        Value(data, properties).validate()

        properties["@schema"]["@columns"]["1"] = [{"@type": "Integer", "@name": "s"}]
        with self.assertRaisesRegex(TypeError, "type does not match"):
            Value(data, properties).validate()
        properties["@schema"]["@columns"]["1"].append({"@type": "String", "@name": "s"})
        Value(data, properties).validate()

    @pytest.mark.xfail(reason="Table in Table is not fully supported at the moment.")
    def test_empty_table_issues(self):
        """Test corner cases with table and subtable."""
        dtype = [("A - String", numpy.object), ("B - Table", numpy.object)]
        properties = {
            "@columns": {"0": [{"@type": "String"}], "1": [{"@type": "Table"}]}
        }

        props = {
            "@schema": properties,
            "@columns": {
                k: {"@schema": v[0], "@name": k}
                for k, v in properties["@columns"].items()
            },
        }

        # empty table
        data = numpy.array([], dtype=dtype)
        t1 = Value(data, props)
        t1.validate()

        # 1-row table with empty subtable
        empty_subtable = numpy.ma.masked_array(
            data=[],
            dtype=[("sub 0 - Integer", numpy.int8), ("sub 1 - String", numpy.object)],
        )

        data = numpy.array([("first row", empty_subtable)], dtype=dtype)
        t2 = Value(data, props)
        t2.validate()  # TODO here is the fail

        # 1-row table with 1-row subtable
        subtable = numpy.ma.masked_array(
            data=[(123, "data")],
            dtype=[("sub 0 - Integer", numpy.int8), ("sub 1 - String", numpy.object)],
        )
        data = numpy.array([("first row", subtable)], dtype=dtype)
        t3 = Value(data, props)
        t3.validate()

    @staticmethod
    def generate_table_data(rows):
        """Generate complex table data for tests.

        Args:
            rows: number of rows to generate

        Return:
            raw_data, dtype, properties: generated data, its dtype and
                                         properties

        """

        def generate_row(idx, level=0):
            real_matrix = numpy.random.random((idx * 2, idx))

            struct_value = {
                "data": 3.14,
                "time": datetime.now().replace(microsecond=0),
                "count-🐙": 42,
            }

            if level < 0:
                table = [generate_row(i, level + 1) for i in range(idx)]
            else:
                table = [(i, f"row#{i}") for i in range(idx)]
                dtype = [("0", numpy.int64), ("1", numpy.object)]
                table = numpy.ma.masked_array(table, dtype=dtype)

            return (
                # binary
                bytearray(f"byte\x00array\x00#{idx} " + "$" * idx, "utf-8"),
                # boolean
                idx % 2 == 0,
                # dictionary
                {str(i): f"{i} " * i for i in range(idx)},
                # enumeration
                f"foo-{idx} " + "@" * idx,
                # integer
                idx,
                # list
                list(range(idx, idx * 2)),
                # matrix
                numpy.ma.masked_array(real_matrix, mask=real_matrix > 0.5),
                # null
                None,
                # path
                f"./foo-{idx}.txt" + "~" * idx,
                # real
                idx * 1e-5,
                # slice
                slice(idx, idx * 2, 2),
                # string
                " ".join(str(idx)),
                # structure
                struct_value,
                # subset
                [[f"foo-{i} " + "@" * i, f"bar-{i} " + "@" * i] for i in range(idx)],
                # table
                table,
                # timestamp
                datetime.now().replace(microsecond=0),
                # value
                Value(idx),
            )

        dtype = [
            ("0 - Binary", numpy.object),
            ("1 - Boolean", numpy.bool),
            ("2 - Dictionary", numpy.object),
            ("3 - Enumeration", numpy.object),
            ("4 - Integer", numpy.int64),
            ("5 - List", numpy.object),
            ("6 - Matrix", numpy.object),
            ("7 - NULL", numpy.object),
            ("8 - Path", numpy.object),
            ("9 - Real", numpy.float64),
            ("10 - Slice", numpy.object),
            ("11 - String", numpy.object),
            ("12 - Structure", numpy.object),
            ("13 - Subset", numpy.object),
            ("14 - Table", numpy.object),
            ("15 - Timestamp", numpy.object),
            ("16 - Value", numpy.object),
        ]
        struct_schema = {
            "data": [{"@type": "Real", "@name": "result", "@init": float("nan")}],
            "time": [{"@type": "Timestamp", "@name": "moment of glory", "@init": -1}],
            "count-🐙": [{"@type": "Integer", "@name": "attempt number", "@init": -1}],
        }
        subtable_schema = {
            "@columns": {
                "0": [{"@type": "Integer", "@name": "sub-column-1", "@init": 0}],
                "1": [{"@type": "String", "@name": "sub-column-2", "@init": ""}],
            },
            # "@type": "Table"
        }
        properties = {
            "@columns": {
                "0": [{"@type": "Binary"}],
                "1": [{"@type": "Boolean"}],
                "2": [{"@type": "Dictionary"}],
                "3": [{"@type": "Enumeration", "@values": ["foo-3", "foo-2"]}],
                "4": [{"@type": "Integer"}],
                "5": [{"@type": "List"}],
                "6": [{"@type": "Matrix"}],
                "7": [{"@type": "NULL"}],
                "8": [{"@type": "Path"}],
                "9": [{"@type": "Real"}],
                "10": [{"@type": "Slice"}],
                "11": [{"@type": "String"}],
                "12": [{"@init": [], "@type": "Structure", "@schema": struct_schema}],
                "13": [{"@type": "Subset", "@values": ["foo-3", "foo-2"]}],
                "14": [
                    {
                        "@init": [[]],
                        "@type": "Table",
                        "@columns": {
                            k: v for k, v in subtable_schema["@columns"].items()
                        },
                    }
                ],
                "15": [{"@type": "Timestamp"}],
                "16": [],
            }
        }

        properties = {
            "@schema": properties,
            "@columns": {
                k: {"@schema": v[0], "@name": k} if v else {"@name": k}
                for k, v in properties["@columns"].items()
            },
        }
        properties["@columns"]["14"]["@columns"] = {
            k: {"@schema": v[0], "@name": k} if v else {"@name": k}
            for k, v in subtable_schema["@columns"].items()
        }

        raw_data = [generate_row(i) for i in range(0, rows)]
        return raw_data, dtype, properties

    def test_value_in_a_cell(self):
        """Quick test for Table capability to have a Value in a cell."""

        data = numpy.array(
            [
                [3.14, "String", ["l", "i", "s", "t"], Value(123)],
                [4.15, "Strong", ["t", "u", "p", "l", "e"], Value([3, 2, 1])],
            ]
        )
        value = Value(data)
        self.assertEqual(value.type, table.Type)
        self.assertEqual(value.data.ncols, 4)
        self.assertEqual(value.data.nrows, 2)
        self.assertEqual(value.validate(), [])

        with TempPath() as path:
            with file.File(path.path, "w") as f:
                f.write(value=value)
                v = f.read()
                # check the cell containing Values while the file is still open
                self.assertEqual(v[0][3].native, 123)
                self.assertEqual(v[1][3].native, [3, 2, 1])

        # check the cell containing Values after the file was closed
        self.assertEqual(v[0][3].native, 123)
        self.assertEqual(v[1][3].native, [3, 2, 1])

    def test_all_column_types_serialization(self):
        """Test serialization works for all possible types of columns."""

        raw_data, dtype, properties = self.generate_table_data(5)

        def test_read_write(raw_data):
            data = numpy.array(raw_data, dtype=dtype)

            v = Value(data, properties)

            with TempPath() as path:
                with file.File(path.path, "w") as f:
                    value_id = f.write(value=v)

                with file.File(path.path, "r") as f:
                    vv = f.get(value_id)
                    # detach Value from the file
                    vv = Value(vv.native, vv.properties)

                self.assertValuesEqual(v, vv)

            assert len(json.dumps(v.properties)) > len(
                json.dumps(v.type.compress_properties(v.properties))
            )

        # test for some known issues ocurred with HDF-serialization
        # #1 test on empty table
        test_read_write([])
        # #2 test if there is exactly one row
        test_read_write(raw_data[:1])
        # #3 test if there is exactly two rows and they are the same
        test_read_write(raw_data[:1] + raw_data[:1])
        # test on multiple different rows
        test_read_write(raw_data)

    def test_serialization_format(self):
        """Test serialization format for Table.

        This test covers most of all other serialization cases as the
        test table contains rows of all possible types.
        """

        n_rows = 5
        raw_data, dtype, properties = self.generate_table_data(n_rows)
        data = numpy.array(raw_data, dtype=dtype)
        value = Value(data, properties)

        with TempPath() as path:
            with file.File(path.path, "w") as f:
                value_id = f.write(value=value)

            with h5py.File(path.path, "r") as f:
                self.assertEqual(
                    list(f.keys()), [str(value_id)]
                )  # , "File must have only one id"
                g = f[str(value_id)]
                assert isinstance(g, h5py.Group), "Root must be a group"
                assert g["nrows"][()] == n_rows, "Wrong rows count"
                expected_columns = set([str(i) for i in range(17)] + ["nrows"])
                assert expected_columns == set(g), "Wrong set of columns"
                assert g.attrs["@type"] == "Table", "Wrong value type in attributes"
                p = json.loads(g.attrs["@properties"])
                assert p["@schema"]["@type"] == "Table", "Incorrect type in properties"

                stored_as_datasets = {
                    "0": "Binary",
                    "1": "Boolean",
                    "3": "Enumeration",
                    "4": "Integer",
                    "7": "Null",
                    "8": "Path",
                    "9": "Real",
                    "10": "Slice",
                    "11": "String",
                    "15": "Timestamp",
                }
                for column, typename in stored_as_datasets.items():
                    assert isinstance(
                        g[column], h5py.Dataset
                    ), f"{typename} must be stored as a dataset"
                    assert g[column].shape[0] == n_rows, "Wrong shape for a dataset"

                stored_as_groups = {
                    "2": "Dictionary",
                    "5": "List",
                    "6": "Matrix",
                    "12": "Structure",
                    "13": "Subset",
                    "14": "Table",
                    "16": "Value",
                }
                for column, typename in stored_as_groups.items():
                    assert isinstance(
                        g[column], h5py.Group
                    ), f"{typename} must be stored as a group"
                    assert len(g[column].keys()), "Wrong number of items"

    def test_row_deletion(self):
        raw_data = [
            (1, "first row"),
            (2, "second row"),
            (3, "third row"),
            (4, "forth row"),
            (5, "fifth row"),
        ]
        dtype = [("sub 0 - Integer", numpy.int64), ("sub 1 - String", numpy.object)]
        mask = [
            (False, False),
            (False, True),
            (False, False),
            (True, False),
            (False, False),
        ]
        table = numpy.ma.masked_array(data=raw_data, dtype=dtype, mask=mask)
        value = Value(table)

        with TempPath() as path:
            with file.File(path.path, "w") as f:
                f.write(value=value)

            assert value.data[2].data == numpy.array(raw_data[2], dtype=dtype)
            value.data.pop(2)
            assert value.data[2].data == numpy.array(raw_data[3], dtype=dtype)

            with file.File(path.path, "w") as f:
                read_value = f.get()
                assert read_value.data[2].data == numpy.array(raw_data[2], dtype=dtype)
                read_value.data.pop(2)
                assert read_value.data[2].data == numpy.array(raw_data[3], dtype=dtype)

            with file.File(path.path, "w") as f:
                read_value = f.get()
                assert read_value.data[2].data == numpy.array(raw_data[3], dtype=dtype)

                mask_dtype = [
                    ("sub 0 - Integer", numpy.bool),
                    ("sub 1 - String", numpy.bool),
                ]
                assert (
                    read_value.data.mask
                    == numpy.array(mask[:2] + mask[3:], dtype=mask_dtype)
                ).all()

    @staticmethod
    def _check_column_rename(value):
        """Helper method to perform column renaming checks."""
        assert value.type is table.Type

        assert value.data.dtype.names == ("i", "str")
        assert value.data.column_properties(0)["@name"] == "i"
        assert value.data.column_properties(1)["@name"] == "str"
        assert value.data[1].dtype.names == ("i", "str")

        # check basic column rename
        value.data.column_rename("i", "index")
        assert value.data.dtype.names[0] == "index"
        assert value.data.column_properties(0)["@name"] == "index"

        value.data.column_rename(1, "name")
        assert value.data.dtype.names[1] == "name"
        assert value.data.column_properties(1)["@name"] == "name"
        assert value.data[1].dtype.names == ("index", "name")

        value.data.append([0, "NAN"], [False, True])
        value.data.column_rename("name", "value")
        assert value.data[3, "index"] == 0
        assert isinstance(value.data[3, "value"], numpy.ma.core.MaskedConstant)
        assert value.data[3].dtype.names == ("index", "value")
        assert value.data[3].mask.dtype.names == ("index", "value")

    def test_column_rename(self):
        """Check renaming of table columns."""

        dtype = [("i", numpy.int64), ("str", numpy.object)]
        data = numpy.array([(1, "one"), (2, "two"), (3, "three")], dtype=dtype)
        value = Value(data)
        self._check_column_rename(value)

        # now the same check but for value stored in file
        value = Value(data)
        with TempPath() as path:
            with file.File(path.path, "w") as f:
                f.write(value=value)

                value = f.get()
                self._check_column_rename(value)
