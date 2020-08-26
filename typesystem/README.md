# Typesystem: python module for working with pSeven specific data.

This python module is designed for simplify and unify work with ports and
analisys data inside pSeven. It is main goal is to provide convenient access to
data via native python and numpy types along with support for large data
objects. The main concept is to use additional properties to specify exact data
type and optional requirements. Properties are also used for data validation and
for constructing special gui for input and editing of values. Typesystem uses
own binary format for data transfer and storage. Properties, binary format, and
json-based protocol for interaction between gui and backend are described in
[specification](https://docs.google.com/document/d/1OTkySFsZY6KjYq1xGAzSnKs_T4QHpdqQxLY5M-2odeI).

## Typical usecases.

It is important keep in mind that ports data should be kept in separate file
from serialized pSeven block. And that all information about value should be
stored along with value (as close as possible).

### Working with ports values.
  * Read all or some values from file to python native types.
  * Load large table header and properties
      with further partial access to data (read and edit).
  * Add new value to existing file.
  * Copy value from input file to output file without reading it
      (binary copy of value and its properties by value id).
  * Edit value in file without any help from block.
      All data for editing (including possible value types and defaults)
      is stored in file along with value itself.

### Working with history.
  * Create history file without data at all (table header only).
  * Add new row to table.
  * Edit table cell (including set or unset missing value).
  * Edit history table properties and columns properties,
      without changing column types or number of columns.
  * Read some continuos rows from history table (windowed access).
  * Import CSV file into history table
      (including creating new history from CSV).
  * Export of history table to CSV file.

## Binary storage format.

### Implemented features.

#### Must have requirements.
  * Store several values in single editable file.
  * Fast partial (windowed) access and lazy-loading
      for large tables and binary blobs.
      Possibility to load and edit properties without access to data.
  * Support for json-serialized properties for each value in file,
      properties are always loaded in memory, without partial access.
  * Properties editing should be fast enough
      and editing time should not depend on data size.
  * Stream-based import of large CSV file with different type of columns
      (including variable size strings).
  * NULL or missing value is a valid state for value or table cell.
  * Possibility of implementation in different languages: python, c++, java.
  * Fast enough implementation, adding new row to table or new value to file
      must not depend on table or file size.
  * Heterogeneous collections support (lists and dictionaries).
  * Effective storage for large matrices and tables,
      data overhead size must not depend on data size.

#### Desirable requirements.
  * Fast moving of values between files, combining different files into one
      and splitting one file to several files without loading data,
      just by binary copying of file segments.
  * Minimum overhead for simple and oftenly used values
      such as integer or real scalars.
  * Effective way of storing categorical columns in tables.
  * Possibility to reserve some amount of rows in table
      before filling them with values.

### Not implemented but also desirable features.
  * Fast way of removing rows from table
      (but with fast windowed access after some rows are removed).
  * Filter for tables.

### Alternative libraries disadvantages
    ([sqlite](https://www.sqlite.org/docs.html) and
    [hdf](https://www.hdfgroup.org/HDF5/doc/index.html)).
  * No convenient way for storing properties.
  * Can not combine or split values via binary copying.
  * Large overhead for single scalars.

#### Sqlite specific disadvantages.
  * Slow work with large data comparing to hdf and current python implementation.

#### HDF specific disadvantages.
  * Unknown if it is possible to reserve some rows in table.
  * Unknown if it is possible to effectively store table
      with column of variable length strings.
  * No support for filters.

## Implementation overview.

### File.
File is a values archive with several methods for creating/reading and opening
values. Most pSeven blocks should work with File and never use other typesystem
interfaces such Value or convert.

### Value class.
Value is a main value holder class. It stores value itself, properties and link
to value memory buffer, if needed. Value is responsible for serializing and
validation of properties before storing value itself. It uses binary
serialization protocol for described in
[specification](https://docs.google.com/document/d/1OTkySFsZY6KjYq1xGAzSnKs_T4QHpdqQxLY5M-2odeI).
Value also can validate underlying data, return its native representation or
convert to another type (not implemented now). Along with binary serialization
utilities Value class uses Type class for all operations with data.

### Properties class.
Properties is a dictionary with ability to serialize binary into memory buffer
with possibility to reserve some buffer for future expansion of properties size.
It is stored as json serialized to string. Reserve mechanism is needed when
someone wants to change properties without altering other data in file
(which may be large).

### Type classes.
Each type implemented in module with corresponding name inside `.types`
directory. Type module must have class Type derived from the base type class
`.types.common.TypeBase` or implement all required methods and properties in
another way. Type class is used for serialization, validation and verification
of value properties via
[json-schema](http://json-schema.org/documentation.html). All type classes
automatically collected in `.constants` module and accessible by name via
`.constants.TYPES` collection. For detailed description of Type class properties
and methods see `.types.common.TypeBase` class comments.
When implementing new type. First of all create corresponding module, implement
`Type` class and import module in `.types.__init__` module. After this type will
be automatically added to `.constants.TYPES` and accessible in file or value.
Additional changes needed for automatically resolving type from python native
value in `.value.resolve_type` method. Also if conversions to another types are
required they must be implemented in `.convert` module. Storing new type into
table require special treatment too, see `.types.table.Table` implementation for
details.

### Conversion between types.
It is possible to convert value of one type to another, by calling
`.convert.convert` function and specifying desired type schema. pSeven block api
can use this function for exposing value of desired type to block writer, in
order to avoid unnecessary type checks.

### Memory utilities.
All Value, Type, and File classes are working with serialized data via specific
memory manager interface. Normally value underlying are in file may change
during some operations with another values in the same file. In order to prevent
losing exact point in buffer where value is stored each value may have memory
manager object and its id inside it. By this id it is always possible to get
pointer to buffer, offset and size of value segment inside. All operations with
underlying buffer should start with the call to memory manager.

### Schema module.
A set of functions for constructing json-schema for properties (and future
jsons) validation.
