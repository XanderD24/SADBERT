"""
Thin setup.py shim for backwards compatibility with older pip versions
and editable installs (pip install -e .).

All project metadata lives in pyproject.toml.
"""
from setuptools import setup

setup()
