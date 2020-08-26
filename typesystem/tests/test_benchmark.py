# Copyright (C) DATADVANCE, 2010-2020

"""HDF5 serialization benchmark tests."""

import os
from timeit import default_timer as timer

import numpy

from ..file import File
from ..value import Value
from .base_testcase import BaseTestCase


def bench_write(values):
    """Benchmark single type."""

    filename = "tmp.p7file"
    test_start = timer()
    if os.path.exists(filename):
        os.unlink(filename)
    with File(filename, "w") as file:
        for value in values:
            file.write(value=value)
    elapsed = timer() - test_start
    size = os.path.getsize(filename)
    return elapsed, size


class TestGeneral(BaseTestCase):
    """Benchmark for typesystem and its serialization."""

    def test_benchmark_write(self):
        """Benchmark write performance for various types of data.

        Results are intended to be interpreted manually.
        """

        levels = [1, 10, 100, 1000]

        print("\n\tIntegers")
        v = Value(123)
        for N in levels:
            elapsed, size = bench_write([v] * N)
            print(f"{v.type.NAME} x {N:<5d}: {elapsed:#.6f}s, {size:7d} bytes")

        print("\n\tReals")
        v = Value(123.567)
        for N in levels:
            elapsed, size = bench_write([v] * N)
            print(f"{v.type.NAME} x {N:<5d}: {elapsed:.6f}s, {size:7d} bytes")

        print("\n\tStrings")
        v = Value("The quick brown fox jumps over the lazy dog")
        for N in levels:
            elapsed, size = bench_write([v] * N)
            print(f"{v.type.NAME} x {N:<5d}: {elapsed:.6f}s, {size:7d} bytes")

        print("\n\tMatrix")
        for N in levels:
            v = Value(numpy.random.random((N, N)))
            elapsed, size = bench_write([v])
            print(f"{v.type.NAME} {N:<5d} x {N:<5d}: {elapsed:.6f}s, {size:7d} bytes")

        print("\n\tList of integers")
        for N in levels:
            v = Value(range(N))
            elapsed, size = bench_write([v])
            print(
                f"{v.type.NAME} of Integer x {N:<5d}': {elapsed:.6f}s, {size:7d} bytes"
            )

        print("\n\tNested lists")
        v = Value([])
        for i, N in enumerate(levels):
            elapsed, size = bench_write([v])
            print(
                f"{v.type.NAME} level {i} - {str(v.native):10s}: {elapsed:.6f}s, {size:7d} bytes"
            )
            v = Value([v])

        print("\n\tDictionary integer => string")
        for N in levels:
            v = Value({str(i): str(i) for i in range(N)})
            elapsed, size = bench_write([v])
            print(
                f"{v.type.NAME} Integer => String {N:<5d} items': {elapsed:.6f}s, {size:7d} bytes"
            )

        print("\n\tNested dictionaries")
        v = Value({})
        for i, N in enumerate(levels):
            elapsed, size = bench_write([v])
            print(
                f"{v.type.NAME} level {i} - {str(v.native):10s}: {elapsed:.6f}s, {size:7d} bytes"
            )
            v = Value({"0": v.native})

        print("\n\tTable of Integers")
        for N in levels:
            data = [(i,) for i in range(N)]
            dtype = [("First column", numpy.int64)]
            properties = {"@columns": {"0": [{"@type": "Integer"}]}}
            data = numpy.array(data, dtype=dtype)
            v = Value(data, {"@schema": properties})
            elapsed, size = bench_write([v])
            print(
                f"{v.type.NAME} x 1 Column of {N:<5d} Integers: {elapsed:.6f}s, {size:7d} bytes"
            )

        print("\n\tTable of Reals")
        for N in levels:
            data = [(float(i)) for i in range(N)]
            dtype = [("Second column", numpy.float)]
            properties = {"@columns": {"0": [{"@type": "Real"}]}}
            data = numpy.array(data, dtype=dtype)
            v = Value(data, {"@schema": properties})
            elapsed, size = bench_write([v])
            print(
                f"{v.type.NAME} x 1 Column of {N:<5d} Reals: {elapsed:.6f}s, {size:7d} bytes"
            )
