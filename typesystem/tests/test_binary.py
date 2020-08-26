# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem binary type tests."""

import time
import uuid

import numpy
import pytest

from .. import file
from ..value import Value
from .base_testcase import BaseTestCase, TempPath


class TestGeneral(BaseTestCase):
    def _random_bytes(self, size):
        return bytearray(numpy.random.randint(256, size=size).tolist())

    def test_read_write(self):
        self._read_write(self._random_bytes(1397))

    def test_convert(self):
        data = self._random_bytes(42)
        self._test_convert(
            [
                (
                    data,
                    {
                        "Binary": data,
                        "NULL": None,
                        "List": list(data),
                        "Matrix": numpy.array([list(data)], dtype=numpy.int64),
                    },
                )
            ]
        )

    def test_set_value(self):
        values = [(uuid.uuid4(), self._random_bytes(size)) for size in [354, 893, 487]]
        data = self._random_bytes(367)
        with TempPath() as path:
            f = file.File(path.path, "w")
            for k, v in values:
                f.write(k, v)
            with f:
                for k, v in values:
                    self.assertEqual(f[k], v)
                f[values[1][0]] = data
                self.assertEqual(f[values[1][0]], data)
            with file.File(path.path, "w") as f:
                self.assertEqual(f[values[0][0]], values[0][1])
                self.assertEqual(f[values[1][0]], data)
                self.assertEqual(f[values[2][0]], values[2][1])
                f[values[1][0]] = values[1][1]
                self.assertEqual(f[values[1][0]], values[1][1])
            with file.File(path.path, "r") as f:
                for k, v in values:
                    self.assertEqual(f[k], v)

    def test_large_data(self):
        """Test serialization of a big (1GB) binary value.

        @cmake_labels long
        """
        start = (uuid.uuid4(), self._random_bytes(837))
        end = (uuid.uuid4(), self._random_bytes(673))
        id = uuid.uuid4()
        DATA = self._random_bytes(16 * 1024 * 1024)
        NCHUNKS = int(1024 / 16)
        total_size = "%.2f GB" % (len(DATA) * NCHUNKS / (1024 ** 3))
        with TempPath() as path1:
            f = file.File(path1.path, "w")
            f.write(*start)
            f.write(id, bytearray())
            f.write(*end)
            tm = time.time()
            with f:
                value = f.get(id)
                for _ in range(NCHUNKS):
                    value.data.append(DATA)
                print("%s append finished in %f" % (total_size, time.time() - tm))
                self.assertEqual(f[start[0]], start[1])
                self.assertEqual(f[end[0]], end[1])
                tm = time.time()
            with f:
                value = f[id]
                self.assertEqual(len(value), len(DATA) * NCHUNKS)
                print("%s read finished in %f" % (total_size, time.time() - tm))
                with TempPath() as path2:
                    tm = time.time()
                    with file.File(path2.path, "w") as f2:
                        f2.write(*start)
                        f2.write(id, f.get(id))
                        f2.write(*end)
                        print("%s copy finished in %f" % (total_size, time.time() - tm))
                        self.assertEqual(f2[start[0]], start[1])
                        self.assertEqual(f2[end[0]], end[1])
                        self.assertEqual(len(f2.get(id).data), len(DATA) * NCHUNKS)

    def test_validation(self):
        """Test validations method for binary values."""
        properties = {"@schema": {"@min_length": 3, "@max_length": 7}}
        Value(b"asdfs", properties).validate()
        Value(b"asd", properties).validate()
        Value(b"asdfghj", properties).validate()
        with self.assertRaisesRegex(ValueError, "lower"):
            Value(b"a", properties).validate()
        errors = Value(b"a", properties).validate(raise_on_error=False)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        with self.assertRaisesRegex(ValueError, "greater"):
            Value(b"asdfghjk", properties).validate()
        errors = Value(b"asdfghjk", properties).validate(raise_on_error=False)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)

        with TempPath() as path:
            with file.File(path.path, "w") as f:
                f.write(value=b"asdfs", properties=properties)
                v = f.get()
                v.validate()
                v.native = b"asd"
                v.validate()
                v.native = b"asdfghj"
                v.validate()
                v.native = b"a"
                with self.assertRaisesRegex(ValueError, "lower"):
                    v.validate()
                errors = v.validate(raise_on_error=False)
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], ValueError)
                v.native = b"asdfghjk"
                with self.assertRaisesRegex(ValueError, "greater"):
                    v.validate()
                errors = v.validate(raise_on_error=False)
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], ValueError)

    def test_append_read(self):
        """Test read() method of the Binary type."""
        value = Value(b"")
        with pytest.raises(OSError, match="data is larger than allowed"):
            value.data.read(0, 1)
        value.data.append(b"test data")
        assert value.data.read(1, 3) == b"est"
