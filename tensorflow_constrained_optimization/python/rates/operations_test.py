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
"""Tests for operations.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import operations
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class OperationsTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for `Expression`-manipulation functions."""

  def _evaluate_expression(self, expression, extra_update_ops_fn=None):
    """Evaluates and returns both portions of an Expression.

    Args:
      expression: `Expression` to evaluate.
      extra_update_ops_fn: function that takes an `EvaluationMemoizer`, and
        returns a list of ops to execute before evaluation.

    Returns:
      A pair (penalty,constraint) containing the values of the penalty and
      constraint portions of the `Expression`.
    """
    memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.compat.v2.Variable(0, dtype=tf.int32)
    }

    penalty_value, penalty_variables = (
        expression.penalty_expression.evaluate(memoizer))
    constraint_value, constraint_variables = (
        expression.constraint_expression.evaluate(memoizer))

    # We need to explicitly create the variables before creating the wrapped
    # session.
    variables = deferred_tensor.DeferredVariableList(
        expression.extra_variables + penalty_variables + constraint_variables)
    for variable in variables:
      variable.create(memoizer)

    def update_ops_fn():
      if not extra_update_ops_fn:
        update_ops = []
      else:
        update_ops = extra_update_ops_fn(memoizer)
      for variable in variables:
        update_ops += variable.update_ops(memoizer)
      return update_ops

    with self.wrapped_session() as session:
      session.run_ops(update_ops_fn)
      return [
          session.run(penalty_value(memoizer)),
          session.run(constraint_value(memoizer))
      ]

  def test_wrap_rate_raises(self):
    """Make sure that wrap_rate() raises if given an `Expression`."""
    penalty_value = 3.1
    constraint_value = 2.7
    penalty_tensor = tf.constant(penalty_value, dtype=tf.float32)
    constraint_tensor = tf.constant(constraint_value, dtype=tf.float32)

    wrapped = operations.wrap_rate(penalty_tensor, constraint_tensor)
    # We should raise a TypeError if either or both of the parameters to
    # wrap_rate() are Expressions.
    with self.assertRaises(TypeError):
      operations.wrap_rate(penalty_tensor, wrapped)
    with self.assertRaises(TypeError):
      operations.wrap_rate(wrapped, constraint_tensor)
    with self.assertRaises(TypeError):
      operations.wrap_rate(wrapped, wrapped)

  def test_wrap_rate_penalty(self):
    """Make sure that wrap_rate() works correctly when given one parameter."""
    penalty_value = 3.1
    penalty_tensor = tf.constant(penalty_value, dtype=tf.float32)

    wrapped = operations.wrap_rate(penalty_tensor)
    actual_penalty_value, actual_constraint_value = self._evaluate_expression(
        wrapped)
    # Both the penalty and the constraint portions of the Expression were
    # initialized to penalty_tensor.
    self.assertNear(penalty_value, actual_penalty_value, err=1e-6)
    self.assertNear(penalty_value, actual_constraint_value, err=1e-6)

  def test_wrap_rate_penalty_and_constraint(self):
    """Make sure that wrap_rate() works correctly when given two parameters."""
    penalty_value = 3.1
    constraint_value = 2.7
    penalty_tensor = tf.constant(penalty_value, dtype=tf.float32)
    constraint_tensor = tf.constant(constraint_value, dtype=tf.float32)

    wrapped = operations.wrap_rate(penalty_tensor, constraint_tensor)
    wrapped = operations.wrap_rate(penalty_tensor)
    actual_penalty_value, actual_constraint_value = self._evaluate_expression(
        wrapped)
    # The penalty and constraint portions of the Expression were initialized to
    # penalty_tensor and constraint_tensor, respectively.
    self.assertNear(penalty_value, actual_penalty_value, err=1e-6)
    self.assertNear(penalty_value, actual_constraint_value, err=1e-6)

  def test_upper_bound_raises_on_empty_list(self):
    """Make sure that upper_bound() raises for an empty expressions list."""
    with self.assertRaises(ValueError):
      operations.upper_bound([])

  def test_upper_bound_raises_on_tensor(self):
    """Make sure that upper_bound() raises when given a non-Expression."""
    value1 = 3.1
    value2 = 2.7
    tensor1 = tf.constant(value1, dtype=tf.float32)
    expression2 = operations.wrap_rate(tf.constant(value2, dtype=tf.float64))

    # List element is a Tensor, instead of an Expression.
    with self.assertRaises(TypeError):
      operations.upper_bound([tensor1, expression2])

  def test_upper_bound(self):
    """Make sure that upper_bound() creates the correct Expression."""
    values = [0.8, 3.1, -1.6, 2.7]
    bound_value = 1.4
    tensors = [tf.constant(value, dtype=tf.float32) for value in values]
    expressions = [operations.wrap_rate(tt) for tt in tensors]

    bounded = operations.upper_bound(expressions)

    # Before evaluating any expressions, we'll assign "bound_value" to the slack
    # variable, so that we can make sure that the same slack variable is being
    # used for all of the constraints.
    def update_ops_fn(memoizer):
      bound_terms = list(bounded.penalty_expression._terms)
      self.assertEqual(1, len(bound_terms))
      bound_tensor = bound_terms[0].tensor(memoizer)
      return [bound_tensor.assign(bound_value)]

    # Extract the set of constraints, and make sure that there is one for each
    # quantity that is being bounded.
    self.assertEqual(len(values), len(bounded.extra_constraints))
    constraints = list(bounded.extra_constraints)

    # Evaluate the constraint expressions.
    actual_values = []
    for constraint in constraints:
      actual_penalty_value, actual_constraint_value = self._evaluate_expression(
          constraint.expression, update_ops_fn)
      self.assertEqual(actual_penalty_value, actual_constraint_value)
      actual_values.append(actual_penalty_value)
    # Constraints take the form expression <= 0, and these are upper-bound
    # constraints, so we're expecting:
    #     value1 - bound_value <= 0
    #     value2 - bound_value <= 0
    #       ....
    # We sort the constraint expression values since they occur in no particular
    # order.
    actual_values = sorted(actual_values)
    expected_values = sorted(value - bound_value for value in values)
    self.assertAllClose(expected_values, actual_values, rtol=0, atol=1e-6)

  def test_lower_bound_raises_on_empty_list(self):
    """Make sure that lower_bound() raises for an empty expressions list."""
    with self.assertRaises(ValueError):
      operations.lower_bound([])

  def test_lower_bound_raises_on_tensor(self):
    """Make sure that lower_bound() raises when given a non-Expression."""
    value1 = 3.1
    value2 = 2.7
    tensor1 = tf.constant(value1, dtype=tf.float32)
    expression2 = operations.wrap_rate(tf.constant(value2, dtype=tf.float64))

    # List element is a Tensor, instead of an Expression.
    with self.assertRaises(TypeError):
      operations.lower_bound([tensor1, expression2])

  def test_lower_bound(self):
    """Make sure that lower_bound() creates the correct Expression."""
    values = [0.8, 3.1, -1.6, 2.7]
    bound_value = 1.4
    tensors = [tf.constant(value, dtype=tf.float32) for value in values]
    expressions = [operations.wrap_rate(tt) for tt in tensors]

    bounded = operations.lower_bound(expressions)

    # Before evaluating any expressions, we'll assign "bound_value" to the slack
    # variable, so that we can make sure that the same slack variable is being
    # used for all of the constraints.
    def update_ops_fn(memoizer):
      bound_terms = list(bounded.penalty_expression._terms)
      self.assertEqual(1, len(bound_terms))
      bound_tensor = bound_terms[0].tensor(memoizer)
      return [bound_tensor.assign(bound_value)]

    # Extract the set of constraints, and make sure that there is one for each
    # quantity that is being bounded.
    self.assertEqual(len(values), len(bounded.extra_constraints))
    constraints = list(bounded.extra_constraints)

    # Evaluate the constraint expressions.
    actual_values = []
    for constraint in constraints:
      actual_penalty_value, actual_constraint_value = self._evaluate_expression(
          constraint.expression, update_ops_fn)
      self.assertEqual(actual_penalty_value, actual_constraint_value)
      actual_values.append(actual_penalty_value)
    # Constraints take the form expression <= 0, and these are lower-bound
    # constraints, so we're expecting:
    #     bound_value - value1 <= 0
    #     bound_value - value2 <= 0
    #       ....
    # We sort the constraint expression values since they occur in no particular
    # order.
    actual_values = sorted(actual_values)
    expected_values = sorted(bound_value - value for value in values)
    self.assertAllClose(expected_values, actual_values, rtol=0, atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
