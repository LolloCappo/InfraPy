
import infrapy
import numpy as np

# Pytest will discover and run all test functions named `test_*` or `*_test`.

def test_version():
    """ check infrapy exposes a version attribute """
    assert hasattr(infrapy, "__version__")
    assert isinstance(infrapy.__version__, str)


class TestCore:
    """ Testing core functions """

    def test_roi_xy(self):
