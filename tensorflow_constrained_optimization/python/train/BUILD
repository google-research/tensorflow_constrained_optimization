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

licenses(["notice"])  # Apache 2.0

py_library(
    name = "constrained_optimizer",
    srcs = ["constrained_optimizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:constrained_minimization_problem",
    ],
)

py_library(
    name = "lagrangian_optimizer",
    srcs = ["lagrangian_optimizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constrained_optimizer",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "proxy_lagrangian_optimizer",
    srcs = ["proxy_lagrangian_optimizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constrained_optimizer",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:constrained_minimization_problem",
    ],
)

py_test(
    name = "lagrangian_optimizer_test",
    srcs = ["lagrangian_optimizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":lagrangian_optimizer",
        ":test_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "proxy_lagrangian_optimizer_test",
    size = "small",
    timeout = "moderate",
    srcs = ["proxy_lagrangian_optimizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":proxy_lagrangian_optimizer",
        ":test_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)
