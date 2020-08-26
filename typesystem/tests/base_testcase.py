# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem utility classes for tests."""

import json
import os
import tempfile
import unittest

import numpy
import pytest

from .. import file
from ..convert import convert
from ..value import Value


class TempPath(object):
    """Temporary file object, file automatically created on enter
        and deleted on exit
    Attributes:
        path: temporary file path.
    """

    def __init__(self):
        self.path = None

    def __enter__(self):
        self.path = tempfile.mktemp()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.path is not None and os.path.exists(self.path):
            os.unlink(self.path)


class BaseTestCase(unittest.TestCase):
    """Base class for typesystem tests.

    Contains some useful asserts and helper functions.
    """

    def assertEqual(self, lhs, rhs, msg=None):
        """Check equality for numpy structured arrays and masked arrays."""
        if not isinstance(lhs, numpy.ndarray) or not isinstance(rhs, numpy.ndarray):
            super(BaseTestCase, self).assertEqual(lhs, rhs, msg)
            return

        if len(lhs) == 0:
            self.assertEqual(len(rhs), 0)
            return

        if lhs.dtype.names:
            self.assertEqual(lhs.dtype.names, rhs.dtype.names)
            for name in lhs.dtype.names:
                self.assertEqual(lhs[name], rhs[name])
        else:
            self.assertTrue(numpy.all(numpy.array(lhs == rhs)))

        if hasattr(rhs, "mask") and hasattr(lhs, "mask"):
            self.assertTrue(numpy.all(numpy.array(lhs.data == rhs.data)))
            self.assertTrue(numpy.all(numpy.array(lhs.mask == rhs.mask)))
        else:
            self.assertEmptyMask(lhs)
            self.assertEmptyMask(rhs)

    def assertValuesEqual(self, value_a, value_b):
        """Check equality of the given instances of typesystem.Value."""
        v = value_a.native
        r = value_b.native
        if isinstance(v, numpy.ma.MaskedArray) and v.dtype.names:
            if v.dtype != r.dtype:
                msg = []
                for f in v.dtype.fields:
                    expected_type = v.dtype.fields[f][0]
                    actual_type = r.dtype.fields.get(f, ["<not present>"])[0]
                    if expected_type != actual_type:
                        msg.append(
                            f"field {f!r} expected to have {expected_type!r} type but got {actual_type!r}"
                        )
                for f in r.dtype.fields:
                    if f not in v.dtype.fields:
                        msg.append(f"field {f!r} is unexpected")
                msg = "\n".join(msg)
                self.fail(f"dtype mismatch: {msg}")

            for column in v.dtype.names:
                if v.dtype[column] == numpy.dtype(numpy.float):
                    numpy.testing.assert_almost_equal(
                        v[column], r[column], err_msg=f"Column '{column}' mismatch"
                    )
                else:
                    if len(v[column]) and isinstance(
                        v[column][0], numpy.ma.MaskedArray
                    ):
                        try:
                            is_equal = [
                                numpy.ma.allequal(v[column][i], r[column][i])
                                for i in range(len(v[column]))
                            ]
                        except TypeError:
                            # numpy.ma.allequal is unable to calc equality for masked
                            # structured arrays so we do it manually
                            # NOTE: all(v[column].data == r[column].data) may produce
                            #       incorrect results when v[column].data contains
                            #        an empty array
                            is_equal_data = [
                                all(v[column][i].data == r[column][i].data)
                                for i in range(len(v[column]))
                            ]
                            is_equal_mask = [
                                all(v[column][i].mask == r[column][i].mask)
                                for i in range(len(v[column]))
                            ]
                            is_equal = all(is_equal_data) and all(is_equal_mask)
                    elif len(v[column]) and isinstance(v[column][0], Value):
                        is_equal = len(v[column]) == len(r[column])
                        if is_equal:
                            for i in range(len(v[column])):
                                if (
                                    v[column][i].native != r[column][i].native
                                    or v[column][i].properties
                                    != r[column][i].properties
                                ):
                                    is_equal = False
                                    break
                    else:
                        is_equal = v[column] == r[column]
                    try:
                        is_equal = all(is_equal)
                    except TypeError:
                        pass
                    self.assertTrue(is_equal, f"Column '{column}' mismatch")

        elif isinstance(v, (numpy.ndarray,)):
            assert (
                v.dtype == r.dtype
            ), f"dtype mismatch: expected {v.dtype} got {r.dtype}"
            numpy.testing.assert_almost_equal(v, r, err_msg="Value mismatch")
            if isinstance(v, numpy.ma.masked_array):
                numpy.testing.assert_almost_equal(
                    v.mask, r.mask, err_msg="Value mask mismatch"
                )
        else:
            self.assertTrue(v == r, f"Value mismatch: expected {v}, got {r}")
        # we could not compare properties as dicts, because nan value could be present
        # so here is a little workaround
        self.assertTrue(
            json.dumps(value_a.properties, sort_keys=True)
            == json.dumps(value_b.properties, sort_keys=True),
            f"Properties mismatch: {value_a.properties} != {value_b.properties}",
        )
        self.assertTrue(value_a.type == value_b.type, "Type mismatch")

    def assertEmptyMask(self, value, msg=None):
        """Check that value does not have missing cells."""
        if not hasattr(value, "mask"):
            return
        # kludge for single column mask
        if isinstance(value.mask[0], numpy.bool_):
            self.assertFalse(numpy.any(list(value.mask), msg))
        else:
            self.assertFalse(numpy.any([list(r) for r in value.mask]), msg)

    def _read_write(self, value, properties=None, compare=None):
        """Write, read and compare read result for single value.

        Args:
            value: native python value.
            properties: value properties (may be None).
            compare: result comparator function: f(original, result),
                assertEqual is used if None.

        """
        test = self.assertEqual if compare is None else compare
        with TempPath() as path:
            with file.File(path.path, "w") as f:
                f.write(value=value, properties=properties)
                test(value, f.read())
                id = f.ids()[0]
                f[id] = value
                test(value, f[id])
            result = f.read()
            test(value, result)
            return result

    def _test_convert(self, values):
        """Batch test for conversion.

        Args:
            values: list of values to convert and results:
            [(<value-to-convert>, {<type-name>: <conversion-native-result>})*]

        """
        for value, data in values:
            for type, result in data.items():
                converted = convert({"@type": type}, value)
                to_compare = converted
                if isinstance(value, Value):
                    to_compare = converted.native
                self.assertEqual(to_compare, result)

    def _test_convert_error(self, values):
        """Batch test for conversion which must fail.

        Args:
            values: list of values to convert and expected errors:
            [(<value-to-convert>, {<type-name>: <expected exception with message>})*]

        """
        for value, data in values:
            for target_type, expected_error in data.items():
                with pytest.raises(ValueError, match=expected_error):
                    converted = convert({"@type": target_type}, value)
