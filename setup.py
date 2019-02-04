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

from setuptools import find_packages
from setuptools import setup

_version = "0.1"
# TODO: change this to a group email address.
_author_email = "no-reply@google.com"

_install_requires = [
    "numpy",
    "scipy",
    "six",

    # TODO: Uncomment this once TF can automatically select between CPU
    # and GPU installation:
    #   "tensorflow<2",
    #   "tensorflow-gpu<2",
]
# TODO: get rid of this, once _install_requires includes TensorFlow.
_extras_require = {
    "tensorflow": "tensorflow<2",
    "tensorflow-gpu": "tensorflow-gpu<2",
}

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
    name="tensorflow_constrained_optimization",
    version=_version,
    author="Google Inc.",
    author_email=_author_email,
    license="Apache 2.0",
    classifiers=_classifiers,
    install_requires=_install_requires,
    extras_require=_extras_require,
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
