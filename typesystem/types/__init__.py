# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem types list.

Each type must be implemented via class with name 'Type' in corresponding module
Type class is a static class for working with type values,
it should not have any state. Type class is used for creating, loading, writing,
validation and making python-native type from values. Type may use it is own
class as value holder (for lazy-loading of big data) or use python-native type
for data representation.
Base classes for type implementation are placed in .common module.

Type class should have methods and properties of .common.TypeBase class,
it is better to implement Type class by deriving.
from .common.TypeBase or another .common.TypeBase successor.
"""

from . import (
    binary,
    boolean,
    common,
    dictionary,
    enumeration,
    integer,
    list,
    matrix,
    null,
    path,
    real,
    slice,
    string,
    structure,
    subset,
    table,
    timestamp,
)
