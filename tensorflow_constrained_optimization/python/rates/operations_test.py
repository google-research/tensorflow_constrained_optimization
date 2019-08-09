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

from tensorflow_constrained_optimization.python.rates import operations

_DENOMINATOR_LOWER_BOUND_KEY = "denominator_lower_bound"
_GLOBAL_STEP_KEY = "global_step"


class OperationsTest(tf.test.TestCase):
  """Tests for `Expression`-manipulation functions."""

  def _evaluate_expression(self, expression, pre_train_ops=None):
    """Evaluates and returns both portions of an Expression.

    Args:
      expression: `Expression` to evaluate.
      pre_train_ops: collection of `Operation`s to execute before evaluation.

    Returns:
      A pair (penalty,constraint) containing the values of the penalty and
      constraint portions of the `Expression`.
    """
    if not pre_train_ops:
      pre_train_ops = set()
    else:
      pre_train_ops = set(pre_train_ops)

    memoizer = {
        _DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        _GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32)
    }

    penalty_tensor, penalty_pre_train_ops, _ = (
        expression.penalty_expression.evaluate(memoizer))
    pre_train_ops |= penalty_pre_train_ops
    constraint_tensor, constraint_pre_train_ops, _ = (
        expression.constraint_expression.evaluate(memoizer))
    pre_train_ops |= constraint_pre_train_ops

    with self.session() as session:
      session.run(
          [tf.global_variables_initializer(),
           tf.local_variables_initializer()])
      session.run(list(pre_train_ops))
      return session.run([penalty_tensor, constraint_tensor])

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

  def test_upper_bound_raises_on_different_dtypes(self):
    """Make sure that upper_bound() raises when given different dtypes."""
    value1 = 3.1
    value2 = 2.7
    expression1 = operations.wrap_rate(tf.constant(value1, dtype=tf.float32))
    expression2 = operations.wrap_rate(tf.constant(value2, dtype=tf.float64))

    # List elements have different dtypes.
    with self.assertRaises(TypeError):
      operations.upper_bound([expression1, expression2])

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
    bound_tensor = bounded.penalty_expression.tensor
    self.assertEqual(tf.float32, bound_tensor.dtype.base_dtype)
    pre_train_ops = [tf.assign(bound_tensor, bound_value)]

    # Extract the set of constraints, and make sure that there is one for each
    # quantity that is being bounded.
    self.assertEqual(len(values), len(bounded.extra_constraints))
    constraints = list(bounded.extra_constraints)

    # Evaluate the constraint expressions.
    actual_values = []
    for constraint in constraints:
      actual_penalty_value, actual_constraint_value = self._evaluate_expression(
          constraint.expression, pre_train_ops)
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

  def test_lower_bound_raises_on_different_dtypes(self):
    """Make sure that lower_bound() raises when given different dtypes."""
    value1 = 3.1
    value2 = 2.7
    expression1 = operations.wrap_rate(tf.constant(value1, dtype=tf.float32))
    expression2 = operations.wrap_rate(tf.constant(value2, dtype=tf.float64))

    # List elements have different dtypes.
    with self.assertRaises(TypeError):
      operations.lower_bound([expression1, expression2])

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
    bound_tensor = bounded.penalty_expression.tensor
    self.assertEqual(tf.float32, bound_tensor.dtype.base_dtype)
    pre_train_ops = [tf.assign(bound_tensor, bound_value)]

    # Extract the set of constraints, and make sure that there is one for each
    # quantity that is being bounded.
    self.assertEqual(len(values), len(bounded.extra_constraints))
    constraints = list(bounded.extra_constraints)

    # Evaluate the constraint expressions.
    actual_values = []
    for constraint in constraints:
      actual_penalty_value, actual_constraint_value = self._evaluate_expression(
          constraint.expression, pre_train_ops)
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
