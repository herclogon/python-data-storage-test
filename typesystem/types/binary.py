# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem binary type."""

import h5py
import numpy

from .. import schema
from .common import SimpleHDFType


class Type(SimpleHDFType):
    """Binary type implementation.

    All functions expect instance of Binary object as value-object.
    """

    NAME = "Binary"
    NATIVE_TYPE = bytearray

    SCHEMA = schema.STRING_SCHEMA()

    @classmethod
    def read(cls, hdf_file, hdf_path, properties):
        return Binary(None, hdf_file, hdf_path)

    @classmethod
    def write(cls, hdf_file, hdf_path, value, properties):
        value.write(hdf_file, hdf_path)

    @classmethod
    def to_native(cls, value, properties):
        if value._data is not None:
            return value._data
        return value.read(0, len(value))

    @classmethod
    def from_native(cls, value, properties):
        return Binary(value)

    @classmethod
    def validate(cls, value, schema, errors_list):
        super(Type, cls).validate(value, schema, errors_list)
        cls._validate_string_length(value, schema, errors_list)

    @classmethod
    def to_hdf_column(cls, data, column_properties):

        dtype = h5py.special_dtype(vlen=numpy.dtype("uint8"))
        # special kludge for binary type: convert array of bytes to
        # array of variable length arrays of uint8
        # see also: https://github.com/h5py/h5py/issues/1178

        # NOTE: we have to create an empty array first and then to fill
        # it because when Numpy creates an array with numpy.array()
        # it makes some guesses about the array shape and in our case
        # these guesses can be wrong, e.g.:
        #   >>> numpy.array([numpy.array([1,2,3], dtype=numpy.uint8)], dtype=numpy.object)
        #   becomes just
        #     array([[1, 2, 3]], dtype=object)
        #   instead of
        #     array([array([1, 2, 3], dtype=uint8)], dtype=object)
        #   and that breaks saving to the hdf-file
        tmp_data = numpy.empty(shape=(len(data),), dtype=dtype)

        def to_uint8(item):
            if item is None:
                item = b""
            elif isinstance(item, str):
                item = bytes(item, "utf-8")
            return numpy.frombuffer(item, dtype=numpy.uint8)

        tmp_data[:] = [to_uint8(row) for row in data]
        data = tmp_data
        return data, dtype

    @classmethod
    def from_hdf_column(cls, column_data, column_properties):
        # special kludge for binary type: convert array of uint8 to bytes
        # NOTE: we have to create an empty ndarray to have full control
        #       of its type and structure
        tmp_data = numpy.empty(shape=(len(column_data),), dtype=numpy.object)
        tmp_data[:] = [bytes(memoryview(row)) for row in column_data]
        return tmp_data


class Binary(object):
    """Value object proxy for lazy-loading."""

    def __init__(self, data, hdf_file=None, hdf_path=None):
        assert (data is None) != (hdf_file is None) and (hdf_file is None) == (
            hdf_path is None
        )
        self._data = None if data is None else Type.NATIVE_TYPE(data)
        self._hdf_file = hdf_file
        self._hdf_path = hdf_path

    def __len__(self):
        if self._data is not None:
            return len(self._data)
        return len(self._hdf_file[self._hdf_path])

    def read(self, start, size):
        if start + size > len(self):
            raise IOError("data is larger than allowed")

        if self._data is not None:
            return Type.NATIVE_TYPE(self._data[start : start + size])
        ds = self._hdf_file[self._hdf_path]
        # @TODO consider using HDF's read_direct() method
        return bytearray(memoryview(ds[start : start + size]))

    def write(self, hdf_file, hdf_path):
        # as we want our value to be expandable we can't rely on h5py.special_dtype(vlen=str)
        # and have to convert bytes to uint8.
        if self._data is not None:
            dtype = numpy.dtype("uint8")
            data = numpy.frombuffer(self._data, dtype=numpy.uint8)
            hdf_file.create_dataset(
                hdf_path,
                shape=(len(self._data),),
                maxshape=(None,),
                data=data,
                dtype=dtype,
            )
        else:
            # @TODO consider using HDF's copy()/write_direct() method
            dtype = numpy.dtype("uint8")
            ds = self._hdf_file[self._hdf_path]
            hdf_file.create_dataset(
                hdf_path, shape=(len(ds),), maxshape=(None,), data=ds, dtype=dtype
            )

    def append(self, data):
        if not isinstance(data, (bytes, bytearray)):
            data = bytes(data)

        if self._data is not None:
            self._data += data
            return

        ds = self._hdf_file[self._hdf_path]
        N = len(ds)
        ds.resize((N + len(data),))
        # @TODO consider using HDF's write_direct() method
        ds[N : N + len(data)] = numpy.frombuffer(data, dtype=numpy.uint8)
