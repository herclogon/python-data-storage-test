# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem helper classes and functions."""


class ErrorsList(object):
    """ List-like class for accumulating errors in list
        or throw error on first append.
    """

    def __init__(self, raise_on_error):
        self._raise = raise_on_error
        self._list = []

    def append(self, error):
        if self._raise:
            raise error
        self._list.append(error)

    def list(self):
        return self._list
