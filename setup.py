#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup script for synth-cic-ids-2018 minimal package."""

from setuptools import setup, find_packages

setup(
    name="synth-cic-ids-2018",
    version="1.0.0",
    description="Autonomous standalone synthetic dataset generator for CIC-IDS-2018",
    packages=find_packages(include=["core", "core.*", "configs", "configs.*"]),
    include_package_data=True,
    install_requires=[
        "pyspark>=3.5.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.10",
)
