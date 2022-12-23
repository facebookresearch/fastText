#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import sysconfig
import io

from pybind11.setup_helpers import Pybind11Extension, ParallelCompile
from setuptools import setup

ParallelCompile().install()

__version__ = '0.9.2'
FASTTEXT_SRC = "src"

WIN = sys.platform.startswith("win32") and "mingw" not in sysconfig.get_platform()

fasttext_src_files = map(str, os.listdir(FASTTEXT_SRC))
fasttext_src_cc = list(filter(lambda x: x.endswith('.cc'), fasttext_src_files))

fasttext_src_cc = list(
    map(lambda x: str(os.path.join(FASTTEXT_SRC, x)), fasttext_src_cc)
)

extra_compile_args = []
if WIN:
    extra_compile_args.append('/DVERSION_INFO=\\"%s\\"' % __version__)
else:
    extra_compile_args.append('-DVERSION_INFO="%s"' % __version__)
    extra_compile_args.extend(["-O3", "-flto"])


def _get_readme():
    """
    Use pandoc to generate rst from md.
    pandoc --from=markdown --to=rst --output=python/README.rst python/README.md
    """
    with io.open("README.rst", encoding='utf-8') as fid:
        return fid.read()


setup(
    name='fasttext-predict',
    version=__version__,
    author='Alexandre Flament',
    author_email='alex.andre@al-f.net',
    description='fasttext Python bindings, only the predict method, no numpy dependency',
    long_description=_get_readme(),
    ext_modules=[
        Pybind11Extension(
            "fasttext_pybind",
            ["python/fasttext_module/fasttext/pybind/fasttext_pybind.cc"] + fasttext_src_cc,
            include_dirs=[
                FASTTEXT_SRC,
            ],
            cxx_std=11,
            extra_compile_args=extra_compile_args,
        )
    ],
    url='https://github.com/dalf/fasttext-predict/',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    packages=[
        'fasttext',
    ],
    package_dir={
        '': 'python/fasttext_module'
    },
    zip_safe=False,
)
