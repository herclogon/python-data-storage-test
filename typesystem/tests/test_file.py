# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem file with values tests."""

import uuid

from .. import file
from .base_testcase import BaseTestCase, TempPath


class TestGeneral(BaseTestCase):
    def test_write_read(self):
        with TempPath() as path:
            file.File(path.path, "w").write(value=42)
            f = file.File(path.path, "r")
            self.assertEqual(f.read(), 42)
            self.assertEqual(len(f.ids()), 1)
            with file.File(path.path, "w") as f:
                id = f.ids()[0]
                self.assertEqual(f[id], 42)
                self.assertEqual(f.get(id).native, 42)
                self.assertEqual(f.get().native, 42)
                f[id] = 12
                self.assertEqual(f[id], 12)
            self.assertEqual(f.read(), 12)
            with f:
                self.assertEqual(f[id], 12)
            with file.File(path.path, "r") as f:
                self.assertEqual(f[id], 12)

    def test_write_read_complex(self):
        ids = [uuid.uuid4(), uuid.uuid4()]
        values = [42, 142]
        with TempPath() as path:
            with file.File(path.path, "w") as f:
                self.assertEqual(len(f.ids()), 0)
                f.write(ids[0], values[0])
                self.assertEqual(len(f.ids()), 1)
                self.assertIn(ids[0], f.ids())
                self.assertEqual(f[ids[0]], values[0])
                f.write(ids[1], values[1])
                self.assertEqual(len(f.ids()), 2)
                self.assertTrue(all([f[id] == v for id, v in zip(ids, values)]))
            with file.File(path.path, "r") as f:
                self.assertEqual(len(f.ids()), 2)
                self.assertTrue(all([f[id] == v for id, v in zip(ids, values)]))
