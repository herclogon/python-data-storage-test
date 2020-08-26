# Copyright (C) DATADVANCE, 2010-2020

import os
import uuid

import h5py

from .value import Value


class File(object):
    """File object represents HDF-file containing typesystem values.

    On the top level File behaves as a dictionary where key
    is uid and values are typesystem values.

    Supports dictionary-like interface for working with native values.
    (When used as context manager, values may be loaded with lazy
    loading support and low memory consumption.)
    """

    def __init__(self, path, mode="r"):
        super(File, self).__init__()
        self._path = path
        self._mode = mode
        self._hdf = None
        self._file = None
        self._values = {}

    @property
    def path(self):
        """File path."""
        return self._path

    def readonly(self):
        """True if file opened in readonly mode."""
        return self._mode == "r"

    def ids(self):
        """List of values ids stored in file."""
        return self._enter_before_call(
            lambda: list(map(self._id_to_key, self._hdf.keys()))
        )

    def read(self, id=None):
        """Read native value from file by value id.

        Args:
            id: Value id, may be None if in file exactly one value.

        """
        return self._enter_before_call(lambda: self.get(id).native)

    def _write_value_to_path(self, path, value):
        """Write the given value to the given path inside HDF-database.

        Args:
            path: string
            value: typesystem.Value

        """
        assert isinstance(
            value, Value
        ), f"Value should be typesystem.Value' type but is of '{type(value)}' type"
        assert self._hdf is not None, "HDF file must be open"

        value.write(self._hdf, path)

    def write(self, id=None, value=None, properties=None):
        """Append new value to the file.

        Args:
            id: Value id, if omitted id will be generated automatically.
            value: Value to write.
            properties: Value properties.

        """
        if self.readonly():
            raise IOError("file is readonly")
        if id is None:
            id = uuid.uuid4()
        id = self._id_to_key(id)
        if id in self.ids():
            raise ValueError("id already exists")

        to_write = value
        if not isinstance(to_write, Value):
            to_write = Value(to_write, properties or {})

        self._enter_before_call(lambda: self._write_value_to_path(f"/{id}", to_write))
        return id

    def _read_value_from_path(self, path):
        """Read value by the given HDF-path."""
        return Value(hdf_file=self._hdf, hdf_path=path)

    def get(self, id=None):
        """Read value as `Value` object by id.

        Args:
            id: Value id, may be None if in file exactly one value.

        """
        if id is None:
            if len(self) != 1:
                raise ValueError("can not read single id")
            id = self.ids()[0]
        id = self._id_to_key(id)
        if id not in self.ids():
            raise ValueError(
                "unknown value id: '{}', known ids: {}".format(id, self.ids())
            )

        if id in self._values:
            return self._values[id]

        result = self._enter_before_call(lambda: self._read_value_from_path(f"/{id}"))
        self._values[id] = result
        return result

    def _init(self):
        if self._hdf is None:
            if self.readonly():
                openmode = "r"
            else:
                openmode = "r+" if os.path.exists(self._path) else "w"
            self._file = open(self._path, openmode.rstrip("+") + "+b")
            self._hdf = h5py.File(self._file, openmode, libver="latest")

    def close(self):
        self._values = {}
        if self._hdf is not None:
            self._hdf.flush()
            self._hdf.close()
            self._hdf = None
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None

    def __enter__(self):
        if not os.path.exists(self._path):
            if self.readonly():
                raise IOError("unable to create file in readonly mode")

        self._init()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, id):
        return self.get(self._id_to_key(id)).native

    def __setitem__(self, id, value):
        self.get(self._id_to_key(id)).native = value

    def __len__(self):
        return len(self.ids())

    def _enter_before_call(self, function):
        if self._hdf is None:
            with self:
                return function()
        return function()

    def _id_to_key(self, id):
        if isinstance(id, str):
            id = uuid.UUID(id)
        assert isinstance(id, uuid.UUID)
        return id
