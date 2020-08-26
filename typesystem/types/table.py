# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem table type.

Future improvements:
- Support datetime object as table cell value (stored as float64).
"""

import collections
import json

from namedlist import namedlist
import numpy

from . import binary, boolean, integer, real, string
from .. import constants, schema, types
from .common import SimpleHDFType, TypeBase


class Type(TypeBase):
    """Table type implementation.

    All functions expect instance of Table object as value-object.
    """

    NAME = "Table"
    NATIVE_TYPE = numpy.ma.masked_array

    SCHEMA = {
        schema.SchemaKeys.INIT: {
            "type": "array",
            "items": {"type": "array"},
            "default": [[]],
        },
        "@min_rows": schema.UNSIGNED(),
        "@max_rows": schema.UNSIGNED(),
        "@min_cols": schema.UNSIGNED(),
        "@max_cols": schema.UNSIGNED(),
        "@columns": {
            "type": "object",
            "patternProperties": {
                "^\\d+$": schema.PROPERTIES_DEFINITIONS_SCHEMAS_REF()
            },
            "additionalProperties": False,
            "minProperties": 1,
        },
    }

    VALUE_PROPERTIES = {
        "@columns": {
            "type": "object",
            "patternProperties": {
                "^\\d+$": {
                    "type": "object",
                    "properties": {
                        "@schema": schema.PROPERTIES_DEFINITIONS_SCHEMA_REF(),
                        "@name": {"type": "string", "minLength": 1},
                        "@categorical": {"type": "boolean"},
                    },
                    "minProperties": 1,
                }
            },
            "additionalProperties": False,
        }
    }

    @classmethod
    def validate(cls, value, validation_schema, errors_list):
        from ..value import Value

        super(Type, cls).validate(value, validation_schema, errors_list)

        min_rows = validation_schema.get("@min_rows", None)
        if min_rows is not None and len(value) < min_rows:
            errors_list.append(ValueError(f"Number of rows is less than {min_rows}!"))

        max_rows = validation_schema.get("@max_rows", None)
        if max_rows is not None and len(value) > max_rows:
            errors_list.append(ValueError(f"Number of rows is more than {max_rows}!"))

        min_cols = validation_schema.get("@min_cols", None)
        if min_cols is not None and value.ncols < min_cols:
            error_message = f"Number of columns is less than {min_cols}!"
            errors_list.append(ValueError(error_message))

        max_cols = validation_schema.get("@max_cols", None)
        if max_cols is not None and value.ncols > max_cols:
            error_message = f"Number of columns is more than {max_cols}!"
            errors_list.append(ValueError(error_message))

        if "@columns" in validation_schema:
            columns_config = validation_schema["@columns"]
            columns_indices = sorted([int(k) for k in columns_config.keys()])
            schemas = {
                col: {schema.PropertiesKeys.SCHEMAS: columns_config[str(col)]}
                for col in columns_indices
            }
            for idx in range(len(value)):
                row = value.rows(idx, 1, columns_indices)
                for col in columns_indices:
                    # passing properties to the Value constructor is crucial here
                    cell = Value(row[row.dtype.names[col]][0], schemas[col])
                    # pylint: disable=protected-access
                    cell._validate(schemas[col], errors_list)

    @classmethod
    def read(cls, hdf_file, hdf_path, properties):
        return Table(None, properties, hdf_file, hdf_path)

    @classmethod
    def write(cls, hdf_file, hdf_path, value, properties):
        value.write(hdf_file, hdf_path, properties)

    @classmethod
    def to_native(cls, value, properties):
        return value.rows(0, len(value))

    @classmethod
    def from_native(cls, value, properties):
        return Table(value, properties)

    @classmethod
    def update_properties(cls, value, value_properties):
        result = super(Type, cls).update_properties(value, value_properties)
        if value is not None:
            result["@columns"] = {
                str(idx): value.column_properties(idx) for idx in range(value.ncols)
            }

        value_schema = result[schema.PropertiesKeys.SCHEMA]
        if "@columns" in value_schema:
            value_schema["@columns"] = {
                key: Type.update_schemas(schemas)
                for key, schemas in value_schema["@columns"].items()
            }
        return result

    @classmethod
    def compress_properties(cls, value_properties):
        result = super(Type, cls).compress_properties(value_properties)

        value_schema = result.get(schema.PropertiesKeys.SCHEMA, {})
        if "@columns" in value_schema:
            value_schema["@columns"] = {
                key: Type.compress_schemas(schemas)
                for key, schemas in value_schema["@columns"].items()
            }
        return result


# Table column properties.
Column = namedlist(
    "Column", ["name", "flags", "properties", "dtype", "type"], default=None
)


class Table:
    """Value object proxy for lazy-loading.

    May be used as proxy to a Value in typesystem.File or as wrapper around
    numpy structured array.

    Rows data is always returned as structured array.
    """

    def __init__(self, data, value_properties, hdf_file=None, hdf_path=None):
        assert (data is None) != (hdf_file is None)
        assert (hdf_file is None) == (hdf_path is None)
        self._data = data
        self._hdf_file = hdf_file
        self._hdf_path = hdf_path
        self._columns = []
        self._nrows = None
        self._capacity = None

        if self._hdf_file is None:
            self._init_from_data(value_properties)
        else:
            self._init_from_hdf(value_properties)

    def _init_from_data(  # pylint: disable=too-many-branches,too-many-statements
        self, value_properties
    ):
        from .. import value

        # Pylint is not friendly to the named tuples.
        # pylint: disable=no-member

        if self._data.dtype.names is not None and len(self._data.dtype.names) != len(
            self._data.dtype
        ):
            raise ValueError("Not all columns are named!")

        have_named_columns = (
            self._data.dtype.names is not None and len(self._data.dtype.names) != 0
        )
        if not have_named_columns and self._data.ndim == 1:
            if hasattr(self._data, "mask"):
                self._data = numpy.ma.resize(self._data, (1, self._data.shape[0]))
            else:
                self._data.resize((1, self._data.shape[0]))

        self._nrows = len(self._data)
        ncols = len(self._data.dtype)
        if ncols == 0:
            assert self._data.ndim == 2
            ncols = self._data.shape[1]

        columns_config = value_properties.get("@columns", {})
        for idx in range(ncols):
            if len(self._data.dtype):  # pylint: disable=len-as-condition
                dtype = self._data.dtype[idx]
            else:
                dtype = self._data.dtype
            if len(dtype) != 0:  # pylint: disable=len-as-condition
                raise ValueError("Column dtype must be a single type description!")

            column_config = columns_config.get(str(idx), {})
            if have_named_columns and schema.PropertiesKeys.NAME not in column_config:
                column_name = self._data.dtype.names[idx]
                column_config[schema.PropertiesKeys.NAME] = column_name
            if schema.PropertiesKeys.SCHEMA not in column_config:
                column_config[schema.PropertiesKeys.SCHEMA] = {}

            column = Column(properties=column_config, dtype=dtype)
            column_schema = column.properties[schema.PropertiesKeys.SCHEMA]
            type_name = column_schema.get(schema.SchemaKeys.TYPE, None)
            if type_name == value.Value.NAME:
                raise TypeError(
                    f"Schema must not be set for for column type '{type_name}'"
                )
            if dtype.type in [numpy.int8, numpy.int16, numpy.int32, numpy.int64]:
                type_name = type_name or integer.Type.NAME
            elif dtype.type in [numpy.float16, numpy.float32, numpy.float64]:
                type_name = type_name or real.Type.NAME
            elif dtype.type is numpy.bool_:
                type_name = type_name or boolean.Type.NAME
            elif dtype.type is numpy.bytes_:
                type_name = type_name or binary.Type.NAME
            elif dtype.type is numpy.str_:
                type_name = type_name or string.Type.NAME
            else:
                if len(self._data) == 0:  # pylint: disable=len-as-condition
                    if type_name is None:
                        from .. import Value

                        column.type = Value
                    else:
                        column.type = constants.TYPES[type_name]
                else:
                    if have_named_columns:
                        column_values = self._data[self._data.dtype.names[idx]]
                    else:
                        column_values = self._data[:, idx]
                    column.type = value.resolve_type(
                        column_values[0], column.properties
                    )
                    same_types = [
                        value.resolve_type(v, column.properties) is column.type
                        for v in filter(
                            lambda x: x is not numpy.ma.masked, column_values[1:]
                        )
                    ]
                    if not all(same_types):
                        raise NotImplementedError(
                            f"Unable to deduce single type for column {idx}"
                        )

            type_name = column.type.NAME if type_name is None else type_name

            if column.type and column.type.NAME == value.Value.NAME:
                # drop schema for Value-typed column
                del column_config[schema.PropertiesKeys.SCHEMA]

            if have_named_columns:
                column.name = self._data.dtype.names[idx]
            else:
                column.name = "%s[%d]" % (type_name, idx)
            if schema.PropertiesKeys.NAME in column.properties:
                column.name = column.properties[schema.PropertiesKeys.NAME]
            else:
                column.properties[schema.PropertiesKeys.NAME] = column.name

            if column.type is None:
                column_type = constants.TYPES[type_name]
            else:
                column_type = column.type
            column.properties = column_type.update_properties(None, column.properties)
            self._columns.append(column)

        # Convert data to structured array with correct dtype.
        mask = numpy.zeros(
            (self._nrows), dtype=[(name, numpy.bool_) for name in self.dtype.names]
        )
        data = Type.NATIVE_TYPE(numpy.zeros((self._nrows), dtype=self.dtype), mask=mask)
        for col_idx in range(self.ncols):
            if have_named_columns:
                col_data = self._data[self._data.dtype.names[col_idx]]
            else:
                col_data = self._data[:, col_idx]
            data[self.dtype.names[col_idx]] = col_data
            if hasattr(self._data, "mask"):
                if have_named_columns:
                    col_mask = self._data.mask[self._data.dtype.names[col_idx]]
                else:
                    col_mask = self._data.mask[:, col_idx]
                data.mask[self.dtype.names[col_idx]] = col_mask
        self._data = data

    def _init_from_hdf(  # pylint: disable=too-many-branches,too-many-statements
        self, properties
    ):
        node = self._hdf_file[self._hdf_path]
        if "mask" in node:
            mask = node["mask"][()]
        else:
            mask = False
        if "nrows" in node:
            self._nrows = node["nrows"][()]
        else:
            self._nrows = 0
        data = []
        names = []

        if "0" in node:
            self._capacity = len(node["0"])
        else:
            self._capacity = 0

        column_indices = sorted(node.keys() - ["mask", "nrows"])
        if column_indices:
            prop_indices = sorted(properties["@columns"].keys())
            assert column_indices == prop_indices, "Value doesn't match properties"

        if not isinstance(mask, bool):
            column_names = [
                properties["@columns"][str(idx)]["@name"] for idx in column_indices
            ]
            assert column_names == list(
                mask.dtype.names
            ), "Value mask has a wrong dtype"

        for i in range(len(properties["@columns"])):
            column_properties = properties["@columns"][str(i)]
            if schema.PropertiesKeys.SCHEMA in column_properties:
                column_type_name = column_properties[schema.PropertiesKeys.SCHEMA][
                    schema.SchemaKeys.TYPE
                ]
                column_type = constants.TYPES[column_type_name]
            else:
                from ..value import Value

                column_type = Value

            type_name = column_type.NAME

            types_mapping = {
                types.boolean.Type: numpy.bool,
                types.integer.Type: numpy.int64,
                types.real.Type: numpy.float64,
            }

            column = Column(
                name=("%s[%d]" % (type_name, i)),
                dtype=types_mapping.get(column_type, numpy.object),
                properties=column_type.update_properties(None, column_properties),
                type=column_type,
            )
            # pylint:disable=no-member
            if schema.PropertiesKeys.NAME in column.properties:
                column.name = column.properties[schema.PropertiesKeys.NAME]

            self._columns.append(column)

    def write(self, hdf_file, hdf_path, properties):

        if "@columns" in properties:
            # ensure the properties don't contradict the value
            columns_indices = sorted([int(k) for k in properties["@columns"].keys()])
            column_names = [
                properties["@columns"][str(idx)]["@name"] for idx in columns_indices
            ]

            assert len(columns_indices) == self.ncols, "Value doesn't match properties"
            if not isinstance(self.mask, bool):
                assert (
                    list(self.mask.dtype.names) == column_names
                ), "Value mask has a wrong dtype"

        if self._hdf_file:
            self._hdf_file.copy(self._hdf_path, hdf_file)
            return

        node = hdf_file.create_group(hdf_path)
        if any(self.mask):
            mask = self.mask
            node.create_dataset("mask", data=mask, maxshape=(None,))

        if not self.nrows:
            # @kluge due to problems with saving empty arrays of
            # variable length of uint8 we do not store empty row
            # of a table
            # @TODO remove the kluge after https://github.com/h5py/h5py/issues/1253
            #       was fixed
            return

        node.create_dataset("nrows", data=self._nrows, dtype=numpy.int64)

        all_names = self.dtype.names
        for col_idx, column in enumerate(self._columns):
            data = self.rows(columns=[col_idx])[all_names[col_idx]].data
            self._write_column(node, str(col_idx), data, column.properties)

    def _write_column(self, node, col_name, data, column_properties):
        from ..value import Value

        if schema.PropertiesKeys.SCHEMA not in column_properties:
            value_type = Value
        else:
            value_type = constants.TYPES[
                column_properties[schema.PropertiesKeys.SCHEMA][schema.SchemaKeys.TYPE]
            ]

        if issubclass(value_type, SimpleHDFType):
            data, dtype = value_type.to_hdf_column(data, column_properties)
            maxshape = tuple([None, *data.shape[1:]])
            node.create_dataset(col_name, data=data, dtype=dtype, maxshape=maxshape)
        else:
            sg = node.create_group(col_name)
            for i, item in enumerate(data):
                self._write_complex_item(
                    item, column_properties, node, sg.name + f"/{i}"
                )

    def _write_complex_item(self, item, properties, node, node_path):
        """Write value of a complex file to the given hdf-node

        Args:
            item - native value or Value
            properties - value properties
            node - hdf node
            node_path - path inside of node

        """
        if schema.PropertiesKeys.SCHEMA not in properties:
            # write an item from Value-typed column as is
            item.write(node.file, node_path)
        else:
            from .. import Value

            Value(item, properties).write(node.file, node_path)

    def append(self, value, mask=None):
        """Append single row or multiple rows.

        Args:
            value: Value to append.
            mask: Value mask.

        """

        def is_single_row(value):
            try:
                len(value[0])
            except TypeError:
                return True
            return False

        if isinstance(value, numpy.ndarray) and not is_single_row(value):
            self.append_rows(self._masked_rows(value, mask))
        else:
            self.append_rows(self._masked_row(value, mask))

    def append_rows(self, rows, mask=None):
        """Append several rows to the table."""
        rows = self._masked_rows(rows, mask)
        size = len(self)
        self.reserve(size + len(rows))
        self._nrows_set(size + len(rows))
        self._rows_assign(size, rows)

    @property
    def ncols(self):
        """Number of columns."""
        return len(self._columns)

    @property
    def nrows(self):
        """Number of rows."""
        return self._nrows

    @property
    def dtype(self):
        """Data dtype as `numpy.dtype` object."""
        return numpy.dtype([(column.name, column.dtype) for column in self._columns])

    @property
    def mask(self):
        """Copy of whole table mask."""
        if self._hdf_file is None:
            return self.rows(0, len(self)).mask

        self.__assert_file_is_valid()
        node = self._hdf_file[self._hdf_path]
        if "mask" in node:
            mask = node["mask"][()]
        else:
            mask = False

        result = Type.NATIVE_TYPE(
            numpy.zeros(
                shape=(len(self),),
                dtype=[
                    (self.dtype.names[col_idx], self.dtype[col_idx])
                    for col_idx in range(self.ncols)
                ],
            )
        )
        result.mask = mask
        # we should return the mask as a structured array
        return result.mask

    @property
    def capacity(self):
        """Maximum number of rows which could be stored without resize.
        """
        return len(self._data) if self._hdf_file is None else self._capacity

    def reserve(self, nrows):
        """Set desired capacity to at least `nrows`, do nothing if
        capacity is greater or equal than requested value.
        """
        if nrows <= self.capacity:
            return
        capacity = self.capacity
        if capacity == 0:
            capacity = nrows
        while capacity < nrows:
            capacity *= 2
        self._capacity_set(capacity)

    def resize(self, nrows, value=None, mask=None):
        """Set number of rows to specified value, delete rows from the
        end or append rows to the end if necessary.

        Args:
            nrows: New table size.
            value: Row value to append.
            mask: Mask for the `value`.
        """
        if nrows > len(self) and value is None:
            raise NotImplementedError
        if value is not None and len(value) != self.ncols:
            raise ValueError("wrong row default value size")

        self.reserve(nrows)
        size = len(self)
        self._nrows_set(nrows)
        if len(self) > size:
            assert value is not None
            self._rows_assign(
                size,
                numpy.ma.resize(
                    self._masked_row(value, mask), (len(self) - size, self.ncols)
                ),
            )

    def shrink(self):
        """Set capacity to table size and number of reserved rows to
        zero.
        """
        self._capacity_set(len(self))

    def pop(self, idx):
        """Remove row at specified index."""
        if idx < 0 or idx >= len(self):
            raise IndexError
        if self._hdf_file is None:
            self._nrows -= 1
            data = Type.NATIVE_TYPE(numpy.zeros((self._nrows,), dtype=self._data.dtype))
            data[:idx] = self._data[:idx]
            data[idx:] = self._data[idx + 1 :]
            data.mask[:idx] = self._data.mask[:idx]
            data.mask[idx:] = self._data.mask[idx + 1 :]
            self._data = data
            return

        self._nrows -= 1
        node = self._hdf_file[self._hdf_path]
        node["nrows"][()] = self._nrows
        if "mask" in node:
            node["mask"][idx:-1] = node["mask"][idx + 1 :]
        for col_idx in range(self.ncols):
            node[f"{col_idx}"][idx:-1] = node[f"{col_idx}"][idx + 1 :]

    def rows(self, idx=0, size=None, columns=None):
        """Get continuous rows.

        Args:
            idx: Starting row index, if omitted resulting rows are
                started from the beginning.
            size: Number of rows to get, if omitted rows returned from
                the `idx` to the end.
            columns: List of columns to get (may be indices or names),
                if omitted all columns are included in result.
        Returns:
            numpy.ma.masked_array containing requested rows

        """
        if idx < 0:
            idx = len(self) + idx
        size = size or len(self) - idx
        if idx < 0 or size < 0 or len(self) < idx + size:
            raise IndexError

        if columns is None:
            columns = range(self.ncols)
        else:
            columns = [self._column_index(c) for c in columns]
        result = Type.NATIVE_TYPE(
            numpy.zeros(
                shape=(size,),
                dtype=[
                    (self.dtype.names[col_idx], self.dtype[col_idx])
                    for col_idx in columns
                ],
            )
        )
        if self._hdf_file is None:
            for name in result.dtype.names:
                result[name] = self._data[name][idx : idx + size]
            return result

        self.__assert_file_is_valid()
        node = self._hdf_file[self._hdf_path]
        if "mask" not in node:
            mask = False
        else:
            column_names = [self._columns[idx].name for idx in columns]
            mask = node["mask"][idx : idx + size][column_names]

        for col_idx in range(self.ncols):
            if col_idx not in columns:
                continue
            result[self._columns[col_idx].name] = self._read_column(col_idx, idx, size)
        # NOTE: if result is structured (masked) array, then the mask must be structured array too
        # otherwise the assignment below won't have any effect
        result.mask = mask
        return result

    def _read_column(self, col_idx, idx, size):
        """Read a column from HDF5.

        Args:
            cold_idx: column index
            idx: Starting row index, if omitted resulting rows are
                started from the beginning.
            size: Number of rows to get, if omitted rows returned from
                the `idx` to the end.
            columns: List of columns to get (may be indices or names),
                if omitted all columns are included in result.

        Returns:
            read data as a numpy array

        """
        from ..value import Value

        column = self._columns[col_idx]
        node = self._hdf_file[self._hdf_path]
        name = str(col_idx)
        if issubclass(column.type, SimpleHDFType):
            if name not in node:
                # kludge for empty row of a Table
                dtype = numpy.object
                if column.type is types.integer.Type:
                    dtype = numpy.int64
                elif column.type is types.real.Type:
                    dtype = numpy.float64
                column_data = numpy.array([], dtype=dtype)
            else:
                column_data = node[name][idx : idx + size]

            column_data = column.type.from_hdf_column(column_data, column.properties)
        else:
            if name in node:
                subgroup = node[name]
                N = size
            else:
                N = 0
            column_data = numpy.empty(shape=(size,), dtype=numpy.object)
            for i in range(idx, idx + N):
                value = Value(hdf_file=self._hdf_file, hdf_path=subgroup.name + f"/{i}")
                if column.type is not Value:
                    value = value.native
                else:
                    # in Value-type column we must load the value entirely
                    # so we recreate it to unbind it from the file
                    value = Value(value.native, value.properties)
                column_data[i - idx] = value

        return column_data

    def matrix(self, columns=None, idx=0, size=None):
        """Get table data as matrix.

        Args:
            columns: List of columns to get (may be indices or names),
                if omitted all columns are returned.
            idx: Starting row index, if omitted resulting rows are
                started from the beginning.
            size: Number of rows to get, if omitted rows returned from
                the `idx` to the end.

        """
        from ..convert import convert
        from . import matrix

        return convert(
            {schema.SchemaKeys.TYPE: matrix.Type.NAME}, self.rows(idx, size, columns)
        )

    def cell_set(self, row, col, value):
        """Set value to specific cell, cell mask is set to False.

        Args:
            row: Row index.
            col: Column index.
            value: Cell value.

        """
        if row < 0:
            row = len(self) + row
        if row < 0 or len(self) <= row:
            raise IndexError

        col = self._column_index(col)
        if self._hdf_file is None:
            self._data[row][col] = value
            return

        self.__assert_file_is_valid()
        column = self._columns[col]
        node = self._hdf_file[self._hdf_path]
        # TODO if value is ma.masked shouldn't we set mask to True?
        if "mask" in node:
            # NOTE: setting node["mask"][row][col] = False won't have any effect
            row_mask = node["mask"][row]
            row_mask[col] = False
            node["mask"][row] = row_mask
        if column.type is types.binary.Type:
            # TODO kludge - this should be somewhere in binary.Type
            value = numpy.frombuffer(value, dtype=numpy.uint8)
        node[f"{col}"][row] = value

    def column_properties(self, col):
        """Get column properties by column index or name."""
        return self._columns[self._column_index(col)].properties

    def column_rename(self, col, new_name):
        """Rename table column."""
        idx = self._column_index(col)
        column = self._columns[idx]
        if column.name == new_name:
            return
        column.name = new_name
        column.properties[schema.PropertiesKeys.NAME] = new_name

        if self._hdf_file is not None:
            self.__assert_file_is_valid()
            node = self._hdf_file[self._hdf_path]
            properties = json.loads(node.attrs["@properties"])
            properties["@columns"][str(idx)]["@name"] = new_name
            node.attrs["@properties"] = json.dumps(properties)
            if "mask" in node:
                # rename dtypes in the stored mask
                mask = node["mask"][()]
                names = list(mask.dtype.names)
                names[idx] = new_name
                mask.dtype.names = names
                del node["mask"]
                node["mask"] = mask
        else:
            names = list(self._data.dtype.names)
            names[idx] = new_name
            self._data.dtype.names = names
            # workaround for the numpy bug with renaming columns in masked_array
            self._data.mask.dtype.names = names

    def _capacity_set(self, value):
        self._nrows_set(min(value, self._nrows))
        if self._hdf_file is None:
            if hasattr(self._data, "mask"):
                self._data = numpy.ma.resize(self._data, (value,))
            else:
                self._data.resize((value,))
            return

        self.__assert_file_is_valid()
        node = self._hdf_file[self._hdf_path]
        if "mask" in node:
            node["mask"].resize((value,))
        for idx in range(self.ncols):
            name = str(idx)
            if name not in node:
                if issubclass(self._columns[idx].type, SimpleHDFType):
                    # create a dataset for simple types
                    dtype = self._columns[idx].dtype
                    data = numpy.empty((value,), dtype=dtype)
                    self._write_column(node, name, data, self._columns[idx].properties)
                else:
                    # complex types require a group
                    node.create_group(name)
            else:
                # resize only a dataset
                if issubclass(self._columns[idx].type, SimpleHDFType):
                    node[name].resize((value,))

    def _rows_assign(self, idx, values):
        from ..value import Value

        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or len(self) < idx + len(values):
            raise IndexError

        if self._hdf_file is None:
            self._data[idx : idx + len(values)] = values
            return

        self.__assert_file_is_valid()
        node = self._hdf_file[self._hdf_path]
        for col_idx in range(self.ncols):

            column = self._columns[col_idx]
            value_type = self._columns[col_idx].type
            if issubclass(value_type, SimpleHDFType):
                data, dtype = value_type.to_hdf_column(
                    values[self._columns[col_idx].name].data,
                    self._columns[col_idx].properties,
                )

                try:
                    node[str(col_idx)][idx : idx + len(values)] = data
                except TypeError as e:
                    # Fallback for binary type. Although the array shapes are the same,
                    # numpy is unable to do the job correctly:
                    #   (Pdb) node[str(col_idx)][0:1]
                    #   array([array([97, 97], dtype=uint8)], dtype=object)
                    #   (Pdb) data
                    #   array([array([115, 115, 115], dtype=uint8)], dtype=object)
                    #   (Pdb) node[str(col_idx)][0:1] = data
                    #   *** TypeError: Can't broadcast (1, 3) -> (1,)
                    #   (Pdb) node[str(col_idx)][0:1].shape
                    #   (1,)
                    #   (Pdb) data.shape
                    #   (1,)
                    for i in range(len(values)):
                        node[str(col_idx)][idx + i] = data[i]
            else:
                subnode = node[str(col_idx)]
                for i, item in enumerate(values[self._columns[col_idx].name]):
                    self._write_complex_item(
                        item,
                        self._columns[col_idx].properties,
                        node,
                        subnode.name + f"/{idx+i}",
                    )

        if hasattr(values, "mask"):
            if "mask" in node:
                node["mask"][idx : idx + len(values)] = values.mask
            elif any(values.mask):
                mask = self.mask
                node.create_dataset("mask", data=mask, maxshape=(None,))
                node["mask"][idx : idx + len(values)] = values.mask
        else:
            if "mask" in node:
                node["mask"][idx : idx + len(values)] = numpy.zeros(
                    (len(values), self.ncols)
                ).astype(numpy.bool)

    def _nrows_set(self, nrows):
        self._nrows = nrows
        if self._hdf_file is not None:
            self.__assert_file_is_valid()
            node = self._hdf_file[self._hdf_path]
            if "nrows" in node:
                node["nrows"][()] = nrows
            else:
                node.create_dataset("nrows", data=self._nrows, dtype=numpy.int64)

    def _column_index(self, idx_or_name):
        if isinstance(idx_or_name, str):
            for i in range(self.ncols):
                if self._columns[i].name == idx_or_name:
                    return i
            raise KeyError(idx_or_name)
        if idx_or_name < 0:
            idx_or_name = self.ncols + idx_or_name
        if idx_or_name < 0 or self.ncols <= idx_or_name:
            raise IndexError
        return idx_or_name

    def _masked_row(self, value, mask):
        if len(value) != self.ncols:
            raise ValueError("Wrong row size!")
        if not hasattr(value, "mask"):
            value = Type.NATIVE_TYPE(
                [tuple(value)], mask=[tuple([False] * self.ncols)], dtype=self.dtype
            )
        if mask is not None:
            if len(mask) != self.ncols:
                raise ValueError("Wrong mask size!")
            value.mask[0] = tuple(mask)
        if value.dtype != self.dtype:
            raise ValueError("Wrong row dtype!")
        return value

    def _masked_rows(self, value, mask):
        size = len(value)
        if size == 0 or len(value[0]) != self.ncols:
            raise ValueError("Wrong rows shape!")
        if not hasattr(value, "mask"):
            value = Type.NATIVE_TYPE(
                [tuple(r) for r in value],
                mask=[tuple([False] * self.ncols)] * len(value),
                dtype=self.dtype,
            )
        elif value.dtype != self.dtype:
            value = Type.NATIVE_TYPE(value, dtype=self.dtype)
        if mask is not None:
            if len(mask) != len(value):
                raise ValueError("Wrong mask shape!")
            for row_idx, row in enumerate(mask):
                if len(row) != self.ncols:
                    raise ValueError("Wrong mask shape!")
                value.mask[row_idx] = tuple(row)
        if value.dtype != self.dtype:
            raise ValueError("Wrong rows dtype!")
        return value

    def __getitem__(self, idx):
        """Get table row by index or table cell by row index and column
        index or name.
        """
        if isinstance(idx, collections.Iterable):
            if len(idx) != 2:
                raise IndexError
            row = self.rows(idx[0], 1)[0]
            return row[self.dtype.names[self._column_index(idx[1])]]
        return self.rows(idx, 1)[0]

    def __setitem__(self, idx, value):
        """Set value for table row by index or table cell by row index
        and column index or name.
        """
        if isinstance(idx, collections.Iterable):
            if len(idx) != 2:
                raise IndexError
            self.cell_set(idx[0], idx[1], value)
        else:
            self._rows_assign(idx, self._masked_row(value, None))

    def __delitem__(self, idx):
        self.pop(idx)

    def __len__(self):
        return self._nrows

    def __assert_file_is_valid(self):
        if not self._hdf_file:
            raise ValueError("File is closed")
