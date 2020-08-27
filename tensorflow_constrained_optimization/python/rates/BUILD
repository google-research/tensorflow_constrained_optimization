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
        ":defaults",
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
    name = "constraint",
    srcs = ["constraint.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":helpers",
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
        # numpy dep,
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
        ":basic_expression",
        ":constraint",
        ":helpers",
        ":term",
        # six dep,
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
        ":term",
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
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "multiclass_rates",
    srcs = ["multiclass_rates.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":basic_expression",
        ":defaults",
        ":deferred_tensor",
        ":expression",
        ":loss",
        ":predicate",
        ":subsettable_context",
        ":term",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "general_rates",
    srcs = ["general_rates.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":basic_expression",
        ":binary_rates",
        ":defaults",
        ":deferred_tensor",
        ":expression",
        ":multiclass_rates",
        ":subsettable_context",
        ":term",
        # tensorflow dep,
    ],
)

py_library(
    name = "rate_minimization_problem",
    srcs = ["rate_minimization_problem.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constraint",
        ":defaults",
        ":deferred_tensor",
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:constrained_minimization_problem",
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":rate_minimization_problem",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python/train:lagrangian_optimizer",
    ],
)

py_library(
    name = "test_helpers",
    testonly = 1,
    srcs = ["test_helpers.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # six dep,
    ],
)

py_test(
    name = "helpers_test",
    srcs = ["helpers_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":helpers",
        # tensorflow dep,
    ],
)

py_test(
    name = "loss_test",
    srcs = ["loss_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":loss",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "deferred_tensor_test",
    srcs = ["deferred_tensor_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":deferred_tensor",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "predicate_test",
    srcs = ["predicate_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":predicate",
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "subsettable_context_test",
    srcs = ["subsettable_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":subsettable_context",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "term_test",
    srcs = ["term_test.py"],
    python_version = "PY3",
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
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "basic_expression_test",
    srcs = ["basic_expression_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":basic_expression",
        ":defaults",
        ":deferred_tensor",
        ":loss",
        ":predicate",
        ":term",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "expression_test",
    srcs = ["expression_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":basic_expression",
        ":defaults",
        ":deferred_tensor",
        ":expression",
        ":term",
        # six dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "operations_test",
    srcs = ["operations_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":deferred_tensor",
        ":operations",
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "binary_rates_test",
    size = "large",
    srcs = ["binary_rates_test.py"],
    python_version = "PY3",
    # shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":binary_rates",
        ":defaults",
        ":deferred_tensor",
        ":subsettable_context",
        ":test_helpers",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "multiclass_rates_test",
    size = "large",
    srcs = ["multiclass_rates_test.py"],
    python_version = "PY3",
    # shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":deferred_tensor",
        ":multiclass_rates",
        ":subsettable_context",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "general_rates_test",
    timeout = "long",
    srcs = ["general_rates_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":defaults",
        ":deferred_tensor",
        ":general_rates",
        ":subsettable_context",
        ":test_helpers",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "rate_minimization_problem_test",
    srcs = ["rate_minimization_problem_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":binary_rates",
        ":expression",
        ":rate_minimization_problem",
        ":subsettable_context",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)

py_test(
    name = "keras_test",
    srcs = ["keras_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        ":operations",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_constrained_optimization/python:graph_and_eager_test_case",
        # Placeholder for internal dep,
    ],
)
