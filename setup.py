#!/usr/bin/env python3

import setuptools
import os

setuptools.setup(
    name='xvectors',
    packages=['xvectors'],

    install_requires=[
        'torch==1.7.1',
        'scipy==1.6.0',
    ],

    zip_safe=False
)