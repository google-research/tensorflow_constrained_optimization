# Copyright 2018 The TensorFlow Constrained Optimization Authors. All Rights
# Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Package setup script for TensorFlow constrained optimization library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import sys

from setuptools import find_packages
from setuptools import setup

# This version number should always be that of the *next* (unreleased) version.
# Immediately after uploading a package to PyPI, you should increment the
# version number and push to gitHub.
__version__ = "0.2"

if "--release" in sys.argv:
  sys.argv.remove("--release")
  _name = "tensorflow_constrained_optimization"
else:
  # Build a nightly package by default.
  _name = "tfco_nightly"
  __version__ += datetime.datetime.now().strftime(".dev%Y%m%d")

_install_requires = [
    "numpy",
    "scipy",
    "six",
    # For TensorFlow 1.14 and 2.0, this dependency isn't quite correct, since
    # the user might want "tensorflow-gpu" instead of "tensorflow". However, in
    # TensorFlow 1.15 and later (before 2.0), or 2.1 and later, "tensorflow" and
    # "tensorflow-gpu" are no longer distinguished.
    "tensorflow>=1.14",
]

_classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

_description = (
    "A library for performing constrained optimization in TensorFlow")
_long_description = """\
*TensorFlow Constrained Optimization* (TFCO) is a library for optimizing
inequality-constrained problems in TensorFlow.

In the most general case, both the objective function and the constraints are
represented as `Tensor`s, giving users the maximum amount of flexibility in
specifying their optimization problems. Constructing these `Tensor`s can be
cumbersome, so we also provide helper functions to make it easy to construct
constrained optimization problems based on *rates*, i.e. proportions of the
training data on which some event occurs (e.g. the error rate, true positive
rate, recall, etc).
"""

setup(
    name=_name,
    version=__version__,
    author="Google Inc.",
    author_email="no-reply@google.com",
    license="Apache 2.0",
    classifiers=_classifiers,
    install_requires=_install_requires,
    packages=find_packages(),
    include_package_data=True,
    description=_description,
    long_description=_long_description,
    long_description_content_type="text/markdown",
    keywords="tensorflow machine learning optimizer constraint rate",
    url=(
        "https://github.com/google-research/tensorflow_constrained_optimization"
    ),
)
