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
"""Tests for general_rates.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import general_rates
from tensorflow_constrained_optimization.python.rates import subsettable_context
from tensorflow_constrained_optimization.python.rates import test_helpers
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class GeneralRatesTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for general rate-constructing functions."""

  def __init__(self, *args, **kwargs):
    super(GeneralRatesTest, self).__init__(*args, **kwargs)

    self._size = 20

    # We use a fixed fake dataset to make sure that the tests are reproducible.
    # The code for generating this random dataset is:
    #
    #   self._predictions = np.random.randn(self._size)
    #   self._labels = np.random.randint(0, 2, size=self._size) * 2 - 1
    #   self._weights = np.random.rand(self._size)
    #
    # The dataset itself is:
    self._predictions = np.array([
        0.60045541, -0.255384, -0.12057165, -0.54508873, -1.83087048,
        -0.14815863, -0.5394772, -1.03734956, 0.03393562, -0.77458484,
        -1.14221614, -2.20251389, 0.26197907, 1.44996421, 0.97745073,
        -0.40625239, -0.80144011, -0.78522915, 0.23576964, -1.00292122
    ])
    self._labels = np.array(
        [-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1])
    self._weights = np.array([
        0.25273731, 0.74555459, 0.25797331, 0.41320666, 0.81219665, 0.14838222,
        0.29385012, 0.21679816, 0.18076016, 0.12369188, 0.77087854, 0.4620583,
        0.46731194, 0.29395326, 0.00413181, 0.51287413, 0.03472246, 0.07791759,
        0.47179513, 0.33895318
    ])

  @property
  def _context(self):
    """Creates a new non-split and non-subsetted context."""
    # We can't create the context in __init__, since it would then wind up in
    # the wrong TensorFlow graph.
    predictions = tf.constant(self._predictions, dtype=tf.float32)
    labels = tf.constant(self._labels, dtype=tf.float32)
    weights = tf.constant(self._weights, dtype=tf.float32)
    return subsettable_context.rate_context(
        predictions=lambda: predictions,
        labels=lambda: labels,
        weights=lambda: weights)

  def test_precision(self):
    """Checks `precision`."""
    bisection_epsilon = 1e-6
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    expression = general_rates.precision(self._context)

    # Extract the the constraints and the associated variables.
    constraint_list = []
    variables = deferred_tensor.DeferredVariableList()
    for constraint in expression.extra_constraints:
      constraint_value = constraint.expression.constraint_expression.evaluate(
          structure_memoizer)
      constraint_list.append(constraint_value)
      variables += constraint_value.variables
    variables = variables.list
    self.assertEqual(2, len(constraint_list))
    constraints = deferred_tensor.DeferredTensor.apply(
        lambda *args: tf.stack(args), *constraint_list)

    # We need to explicitly create all variables included in the expression
    # before we can try to extract the ratio_bounds.
    for variable in variables:
      variable.create(structure_memoizer)

    # The find_zeros_of_functions() helper will perform a bisection search over
    # the ratio_bounds, so we need to extract the Tensor containing them from
    # the graph.
    ratio_bounds = None
    for variable in variables:
      tensor = variable(structure_memoizer)
      if tensor.name.startswith("tfco_ratio_bounds"):
        self.assertIsNone(ratio_bounds)
        ratio_bounds = tensor
    self.assertIsNotNone(ratio_bounds)

    def update_ops_fn():
      update_ops = []
      for variable in variables:
        update_ops += variable.update_ops(structure_memoizer)
      return update_ops

    with self.wrapped_session() as session:
      session.run_ops(update_ops_fn)

      def evaluate_fn(values):
        """Assigns the variables and evaluates the constraints."""
        session.run_ops(lambda: ratio_bounds.assign(values))
        return session.run(constraints(structure_memoizer))

      actual_ratio_bounds = test_helpers.find_zeros_of_functions(
          2, evaluate_fn, epsilon=bisection_epsilon)
      actual_numerator = actual_ratio_bounds[0]
      actual_denominator = actual_ratio_bounds[1]

    expected_numerator = (
        np.sum((0.5 * (1.0 + np.sign(self._predictions))) *
               (self._labels > 0.0) * self._weights) / np.sum(self._weights))

    expected_denominator = (
        np.sum((0.5 * (1.0 + np.sign(self._predictions))) * self._weights) /
        np.sum(self._weights))

    self.assertAllClose(
        expected_numerator, actual_numerator, rtol=0, atol=bisection_epsilon)
    self.assertAllClose(
        expected_denominator,
        actual_denominator,
        rtol=0,
        atol=bisection_epsilon)

  def test_f_score(self):
    """Checks `f_score`."""
    beta = 1.6
    bisection_epsilon = 1e-6
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    expression = general_rates.f_score(self._context, beta)

    # Extract the the constraints and the associated variables.
    constraint_list = []
    variables = deferred_tensor.DeferredVariableList()
    for constraint in expression.extra_constraints:
      constraint_value = constraint.expression.constraint_expression.evaluate(
          structure_memoizer)
      constraint_list.append(constraint_value)
      variables += constraint_value.variables
    variables = variables.list
    self.assertEqual(2, len(constraint_list))
    constraints = deferred_tensor.DeferredTensor.apply(
        lambda *args: tf.stack(args), *constraint_list)

    # We need to explicitly create all variables included in the expression
    # before we can try to extract the ratio_bounds.
    for variable in variables:
      variable.create(structure_memoizer)

    # The find_zeros_of_functions() helper will perform a bisection search over
    # the ratio_bounds, so we need to extract the Tensor containing them from
    # the graph.
    ratio_bounds = None
    for variable in variables:
      tensor = variable(structure_memoizer)
      if tensor.name.startswith("tfco_ratio_bounds"):
        self.assertIsNone(ratio_bounds)
        ratio_bounds = tensor
    self.assertIsNotNone(ratio_bounds)

    def update_ops_fn():
      update_ops = []
      for variable in variables:
        update_ops += variable.update_ops(structure_memoizer)
      return update_ops

    with self.wrapped_session() as session:
      session.run_ops(update_ops_fn)

      def evaluate_fn(values):
        """Assigns the variables and evaluates the constraints."""
        session.run_ops(lambda: ratio_bounds.assign(values))
        return session.run(constraints(structure_memoizer))

      actual_ratio_bounds = test_helpers.find_zeros_of_functions(
          2, evaluate_fn, epsilon=bisection_epsilon)
      actual_numerator = actual_ratio_bounds[0]
      actual_denominator = actual_ratio_bounds[1]

    expected_numerator = ((1.0 + beta * beta) * np.sum(
        (0.5 * (1.0 + np.sign(self._predictions))) *
        (self._labels > 0.0) * self._weights) / np.sum(self._weights))

    expected_denominator = (((1.0 + beta * beta) * np.sum(
        (0.5 * (1.0 + np.sign(self._predictions))) *
        (self._labels > 0.0) * self._weights) + (beta * beta) * np.sum(
            (0.5 * (1.0 - np.sign(self._predictions))) *
            (self._labels > 0.0) * self._weights) + np.sum(
                (0.5 * (1.0 + np.sign(self._predictions))) *
                (self._labels <= 0.0) * self._weights)) / np.sum(self._weights))

    self.assertAllClose(
        expected_numerator, actual_numerator, rtol=0, atol=bisection_epsilon)
    self.assertAllClose(
        expected_denominator,
        actual_denominator,
        rtol=0,
        atol=bisection_epsilon)


if __name__ == "__main__":
  tf.test.main()
