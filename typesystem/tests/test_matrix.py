# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem matrix type tests."""

import time

import numpy

from .. import Value, file
from .base_testcase import BaseTestCase, TempPath


class TestGeneral(BaseTestCase):
    def test_read_write(self):
        self._read_write(numpy.random.rand(3, 2))
        self._read_write(numpy.random.rand(0, 3))
        self._read_write(numpy.random.rand(0, 0))

    def test_mask_read_write(self):
        for shape in [(5, 11), (0, 4)]:
            data = numpy.ma.masked_array(
                numpy.random.rand(*shape), mask=numpy.zeros(shape, dtype=numpy.bool_)
            )
            if len(data) != 0:
                for r in numpy.random.randint(shape[0], size=2):
                    for c in numpy.random.randint(shape[1], size=2):
                        data.mask[r][c] = True
            self._read_write(data)

    def test_convert(self):
        data = numpy.random.rand(3, 7)
        self._test_convert([(data, {"Matrix": data, "NULL": None})])
        self._test_convert(
            [
                (numpy.array([[1]], dtype=numpy.bool_), {"Boolean": True}),
                (numpy.array([[0]], dtype=numpy.int64), {"Boolean": False}),
                (numpy.array([[42]], dtype=numpy.float64), {"Integer": 42}),
                (numpy.array([[3.14]], dtype=numpy.float64), {"Real": 3.14}),
            ]
        )
        data = numpy.random.rand(1, 5)
        self._test_convert([(data, {"List": data[0].tolist()})])

    def test_int_float_matrix(self):
        """Ensure matrix can be constructed from the mix of int and float values."""
        array_as_list = [[1, 2, 3.14], [4, 5, 6]]
        v = Value(array_as_list, {"@schema": {"@type": "Matrix"}})
        assert (v.native == numpy.array(array_as_list)).all()
        assert v.type.NAME == "Matrix"

    def test_convert_error(self):
        data = numpy.array([3.14, 3.15, 2.71, 2.72])
        self._test_convert_error(
            [
                (
                    data,
                    {
                        "List": "matrix must have exactly one row",
                        "Table": "no way to convert Matrix to Table",
                        "Integer": "matrix size must be exactly 1x1",
                    },
                )
            ]
        )

    def test_from_list(self):
        """Ensure a matrix may be created from an appropriate list of lists."""
        array_as_list = [[1, 2, 3], [4, 5, 6]]
        v = Value(array_as_list, {"@schema": {"@type": "Matrix"}})
        assert (v.native == numpy.array(array_as_list)).all()

    def test_large_matrix(self):
        """
        @cmake_labels long
        """
        data = numpy.random.uniform(0, 1, (100000, 100))
        with TempPath() as path:
            tm = time.time()
            file.File(path.path, "w").write(value=data)
            print("Large matrix write finished in %f" % (time.time() - tm))
            tm = time.time()
            read = file.File(path.path, "r").read()
            print("Large matrix read finished in %f" % (time.time() - tm))
            tm = time.time()
            self.assertEqual(read, data)
            print("Large matrix compare finished in %f" % (time.time() - tm))

    def test_validation(self):
        properties = {"@schema": {"@nature": "symmetric"}}
        Value(numpy.array([[0.0]]), properties).validate()
        Value(numpy.array([[0.0, 1.0], [1.0, 0.0]]), properties).validate()
        with self.assertRaisesRegex(ValueError, "symmetric"):
            Value(numpy.array([[0.0, 1.0], [0.0, 0.0]]), properties).validate()
        with self.assertRaisesRegex(ValueError, "square"):
            Value(numpy.array([[0.0, 1.0]]), properties).validate()

        Value(numpy.array([[0.0, numpy.nan], [numpy.nan, 0.0]]), properties).validate()
        properties["@schema"]["@items"] = {
            "@type": "Real",
            "@name": "r",
            "@minimum": 1.0,
        }
        Value(numpy.array([[3.0, numpy.nan], [numpy.nan, 2.0]]), properties).validate()
        with self.assertRaisesRegex(ValueError, "lower"):
            Value(
                numpy.array([[0.0, numpy.nan], [numpy.nan, 0.0]]), properties
            ).validate()
        properties["@schema"]["@items"]["@nanable"] = False
        with self.assertRaisesRegex(ValueError, "NaN"):
            Value(
                numpy.array([[1.0, numpy.nan], [numpy.nan, 1.0]]), properties
            ).validate()

        value = Value(numpy.array([[1.0, numpy.nan], [numpy.nan, 1.0]]), properties)
        value.data.mask[0][1] = True
        value.data.mask[1][0] = True
        value.validate()
        properties["@schema"]["@allow_missing_items"] = False
        value = Value(numpy.array([[1.0, numpy.nan], [numpy.nan, 1.0]]), properties)
        value.data.mask[0][1] = True
        value.data.mask[1][0] = True
        with self.assertRaisesRegex(ValueError, "missing"):
            value.validate()
