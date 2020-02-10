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
"""Tests for expression.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import expression
from tensorflow_constrained_optimization.python.rates import term
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class ExpressionTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for `Expression` class."""

  def test_arithmetic(self):
    """Tests `Expression`'s arithmetic operators."""
    memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.compat.v2.Variable(0, dtype=tf.int32)
    }

    def constant_expression(penalty_constant, constraint_constant=None):
      penalty_basic_expression = basic_expression.BasicExpression(
          [term.TensorTerm(tf.constant(penalty_constant, dtype=tf.float32))])
      if constraint_constant is None:
        constraint_basic_expression = penalty_basic_expression
      else:
        constraint_basic_expression = basic_expression.BasicExpression([
            term.TensorTerm(tf.constant(constraint_constant, dtype=tf.float32))
        ])
      return expression.Expression(penalty_basic_expression,
                                   constraint_basic_expression)

    # This expression exercises all of the operators.
    expression_object = (
        constant_expression(0.3) - (constant_expression(-3.6, -0.2) / 2.3 +
                                    0.7 * constant_expression(1.5, -0.5)) -
        (constant_expression(1.2) + constant_expression(0.4, 2.3) -
         constant_expression(0.1)) * 0.6 + constant_expression(0.8))

    actual_penalty_value, penalty_variables = (
        expression_object.penalty_expression.evaluate(memoizer))
    actual_constraint_value, constraint_variables = (
        expression_object.constraint_expression.evaluate(memoizer))

    # We need to explicitly create the variables before creating the wrapped
    # session.
    variables = deferred_tensor.DeferredVariableList(penalty_variables +
                                                     constraint_variables)
    for variable in variables:
      variable.create(memoizer)

    # This is the same expression as above, applied directly to the python
    # floats.
    expected_penalty_value = (0.3 - (-3.6 / 2.3 + 0.7 * 1.5) -
                              (1.2 + 0.4 - 0.1) * 0.6 + 0.8)
    expected_constraint_value = (0.3 - (-0.2 / 2.3 + 0.7 * -0.5) -
                                 (1.2 + 2.3 - 0.1) * 0.6 + 0.8)

    with self.wrapped_session() as session:
      print("HELLO")
      print(actual_penalty_value)
      print(actual_penalty_value(memoizer))
      print("GOODBYE")
      self.assertNear(
          expected_penalty_value,
          session.run(actual_penalty_value(memoizer)),
          err=1e-6)
      self.assertNear(
          expected_constraint_value,
          session.run(actual_constraint_value(memoizer)),
          err=1e-6)

  def test_extra_variables(self):
    """Tests that `Expression`s propagate extra variables correctly."""

    def create_dummy_expression(extra_variables=None):
      """Creates an empty `Expression` with the given extra variables."""
      return expression.Expression(
          basic_expression.BasicExpression([]),
          basic_expression.BasicExpression([]),
          extra_variables=extra_variables)

    variable1 = deferred_tensor.DeferredVariable(2.718)
    variable2 = deferred_tensor.DeferredVariable(3.142)
    variable3 = deferred_tensor.DeferredVariable(-1.0)

    expression1 = create_dummy_expression([variable1])
    expression2 = create_dummy_expression([variable2])
    expression3 = create_dummy_expression([variable3])

    expression12 = expression1 * 0.5 + expression2
    expression23 = expression2 - expression3 / 1.3
    expression123 = -expression12 + 0.6 * expression23

    self.assertEqual(expression12.extra_variables, [variable1, variable2])
    self.assertEqual(expression23.extra_variables, [variable2, variable3])
    self.assertEqual(expression123.extra_variables,
                     [variable1, variable2, variable3])

  def test_extra_constraints(self):
    """Tests that `Expression`s propagate extra constraints correctly."""

    def create_dummy_expression(extra_constraints=None):
      """Creates an empty `Expression` with the given extra constraints."""
      return expression.Expression(
          basic_expression.BasicExpression([]),
          basic_expression.BasicExpression([]),
          extra_constraints=extra_constraints)

    constrained_expression = create_dummy_expression()
    constraint1 = (constrained_expression <= 0.0)
    constraint2 = (constrained_expression >= 1.0)
    constraint3 = (constrained_expression <= 2.0)

    expression1 = create_dummy_expression([constraint1])
    expression2 = create_dummy_expression([constraint2])
    expression3 = create_dummy_expression([constraint3])

    expression12 = expression1 * 0.5 + expression2
    expression23 = expression2 - expression3 / 1.3
    expression123 = -expression12 + 0.6 * expression23

    self.assertEqual(expression12.extra_constraints, [constraint1, constraint2])
    self.assertEqual(expression23.extra_constraints, [constraint2, constraint3])
    self.assertEqual(expression123.extra_constraints,
                     [constraint1, constraint2, constraint3])


if __name__ == "__main__":
  tf.test.main()
