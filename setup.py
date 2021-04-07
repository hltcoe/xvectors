#!/usr/bin/env python3

import setuptools

__author__ = 'Alan McCree, Greg Sell, Daniel Garcia Romero, Kiran Karra'
__email__ = 'alan.mccree@jhu.edu,kiran.karra@jhuapl.edu'
__version__ = '0.1'

setuptools.setup(
    name='xvectors',
    version=__version__,

    description='xvector model code',
    url='https://github.com/hltcoe/xvectors',
    python_requires='>=3',
    packages=['xvectors'],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='deep-learning speech xvectors',

    install_requires=[
        'torch==1.7.1',
        'scipy==1.6.0',
        'numpy',
        'kaldi_io',  # note that this is different than the kaldi-io package!
    ],

    scripts=['scripts/train_from_feats.py',
             'scripts/update_model.py'],

    zip_safe=False
)