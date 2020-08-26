# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem null type tests."""

from .base_testcase import BaseTestCase


class TestGeneral(BaseTestCase):
    def test_read_write(self):
        self._read_write(None)

    def test_convert(self):
        self._test_convert([(None, {"NULL": None})])
