# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem value tests."""

import pytest

from .. import File
from ..value import Value
from .base_testcase import BaseTestCase, TempPath


class TestGeneral(BaseTestCase):
    """Test Typesystem.Value."""

    def test_native_value_setter(self):
        """Check native value setter for Value.

        Ensure native value features:
          - can be changed for a standalone Value
          - can be changed for a Value from the File if the File is open
          - can not be changed for a Value from the File if the File is
            closed
        """
        v = Value(123)
        v.native = 321
        assert v.native == 321

        with pytest.raises(
            NotImplementedError, match="Value type and properties mismatch"
        ):
            v.native = "invalid value"

        with TempPath() as path:
            with File(path.path, "w") as f:
                f.write(value=v)
                read_value = f.get()
                read_value.native = 567

            with File(path.path, "w") as f:
                read_value = f.get()
                assert read_value.native == 567

        with pytest.raises(ValueError, match="File is close"):
            read_value.native = 567

        errors = read_value.validate(raise_on_error=True)
        assert len(errors) == 0
