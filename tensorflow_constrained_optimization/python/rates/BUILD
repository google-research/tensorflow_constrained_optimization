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

# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "helpers",
    srcs = ["helpers.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "loss",
    srcs = ["loss.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":helpers",
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "defaults",
    srcs = ["defaults.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":loss",
    ],
)

py_library(
    name = "deferred_tensor",
    srcs = ["deferred_tensor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":helpers",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "predicate",
    srcs = ["predicate.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":deferred_tensor",
        ":helpers",
        # tensorflow dep,
    ],
)

py_library(
    name = "subsettable_context",
    srcs = ["subsettable_context.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":deferred_tensor",
        ":helpers",
        ":predicate",
        # tensorflow dep,
    ],
)

py_library(
    name = "term",
    srcs = ["term.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":deferred_tensor",
        ":helpers",
        ":predicate",
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "basic_expression",
    srcs = ["basic_expression.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":deferred_tensor",
        ":helpers",
    ],
)

py_library(
    name = "expression",
    srcs = ["expression.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":deferred_tensor",
        ":helpers",
        # tensorflow dep,
    ],
)

py_library(
    name = "operations",
    srcs = ["operations.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":basic_expression",
        ":deferred_tensor",
        ":expression",
        ":helpers",
        # tensorflow dep,
    ],
)

py_library(
    name = "binary_rates",
    srcs = ["binary_rates.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":basic_expression",
        ":defaults",
        ":deferred_tensor",
        ":expression",
        ":loss",
        ":subsettable_context",
        ":term",
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "rate_minimization_problem",
    srcs = ["rate_minimization_problem.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:constrained_minimization_problem",
    ],
)

py_test(
    name = "helpers_test",
    srcs = ["helpers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":helpers",
        # tensorflow dep,
    ],
)

py_test(
    name = "loss_test",
    srcs = ["loss_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":loss",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
    ],
)

py_test(
    name = "deferred_tensor_test",
    srcs = ["deferred_tensor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":deferred_tensor",
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
    ],
)

py_test(
    name = "predicate_test",
    srcs = ["predicate_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":predicate",
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
    ],
)

py_test(
    name = "subsettable_context_test",
    srcs = ["subsettable_context_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":subsettable_context",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
    ],
)

py_test(
    name = "term_test",
    srcs = ["term_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":deferred_tensor",
        ":loss",
        ":predicate",
        ":term",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
    ],
)

py_test(
    name = "basic_expression_test",
    srcs = ["basic_expression_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":basic_expression",
        ":defaults",
        ":deferred_tensor",
        ":loss",
        ":predicate",
        ":term",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
    ],
)

py_test(
    name = "expression_test",
    srcs = ["expression_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":basic_expression",
        ":defaults",
        ":deferred_tensor",
        ":expression",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
    ],
)

py_test(
    name = "operations_test",
    srcs = ["operations_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":operations",
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
    ],
)

py_test(
    name = "binary_rates_test",
    size = "large",
    srcs = ["binary_rates_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":binary_rates",
        ":defaults",
        ":deferred_tensor",
        ":subsettable_context",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
    ],
)

py_test(
    name = "rate_minimization_problem_test",
    srcs = ["rate_minimization_problem_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":binary_rates",
        ":deferred_tensor",
        ":rate_minimization_problem",
        ":subsettable_context",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
    ],
)
