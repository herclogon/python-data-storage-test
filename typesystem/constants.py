# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem common constants.

TYPES: dictionary mapping type name to corresponding type class:
    {<type-name>: <Type-class}
"""

import inspect

from . import schema, types


def _types_list():
    return [
        m[1].Type
        for m in inspect.getmembers(types)
        if not m[0].startswith("_") and hasattr(m[1], "Type")
    ]


TYPES = {}
for type in _types_list():
    assert type.NAME not in TYPES
    TYPES[type.NAME] = type

assert len(TYPES) == len(_types_list())

# initialize schema properties definitions here, during first import of types
schema._PROPERTIES_DEFINITIONS = schema._create_properties_definitions(TYPES.values())
