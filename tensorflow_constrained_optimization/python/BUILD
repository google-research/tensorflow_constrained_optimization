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
    name = "candidates",
    srcs = ["candidates.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # scipy dep,
        # six dep,
    ],
)

py_library(
    name = "constrained_minimization_problem",
    srcs = ["constrained_minimization_problem.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
    ],
)

py_library(
    name = "graph_and_eager_test_case",
    testonly = 1,
    srcs = ["graph_and_eager_test_case.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "candidates_test",
    size = "small",
    srcs = ["candidates_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":candidates",
        # numpy dep,
        # tensorflow dep,
    ],
)
