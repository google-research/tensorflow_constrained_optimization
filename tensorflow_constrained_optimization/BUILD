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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_constrained_optimization",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:candidates",
        "//tensorflow_constrained_optimization/python:constrained_minimization_problem",
        "//tensorflow_constrained_optimization/python/rates:binary_rates",
        "//tensorflow_constrained_optimization/python/rates:general_rates",
        "//tensorflow_constrained_optimization/python/rates:keras",
        "//tensorflow_constrained_optimization/python/rates:loss",
        "//tensorflow_constrained_optimization/python/rates:operations",
        "//tensorflow_constrained_optimization/python/rates:rate_minimization_problem",
        "//tensorflow_constrained_optimization/python/rates:subsettable_context",
        "//tensorflow_constrained_optimization/python/train:constrained_optimizer",
        "//tensorflow_constrained_optimization/python/train:lagrangian_optimizer",
        "//tensorflow_constrained_optimization/python/train:proxy_lagrangian_optimizer",
    ],
)
