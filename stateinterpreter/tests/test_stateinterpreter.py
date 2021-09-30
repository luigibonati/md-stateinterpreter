"""
Unit and regression test for the stateinterpreter package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import stateinterpreter


def test_stateinterpreter_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "stateinterpreter" in sys.modules
