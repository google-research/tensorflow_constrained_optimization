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

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v2 as tf

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
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
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
      return expression.ExplicitExpression(penalty_basic_expression,
                                           constraint_basic_expression)

    # This expression exercises all of the operators.
    expression_object = (
        constant_expression(0.3) - (constant_expression(-3.6, -0.2) / 2.3 +
                                    0.7 * constant_expression(1.5, -0.5)) -
        (constant_expression(1.2) + constant_expression(0.4, 2.3) -
         constant_expression(0.1)) * 0.6 + constant_expression(0.8))

    actual_penalty_value = expression_object.penalty_expression.evaluate(
        structure_memoizer)
    actual_constraint_value = expression_object.constraint_expression.evaluate(
        structure_memoizer)

    # We need to explicitly create the variables before creating the wrapped
    # session.
    variables = deferred_tensor.DeferredVariableList(
        actual_penalty_value.variables + actual_constraint_value.variables)
    for variable in variables:
      variable.create(structure_memoizer)

    # This is the same expression as above, applied directly to the python
    # floats.
    expected_penalty_value = (0.3 - (-3.6 / 2.3 + 0.7 * 1.5) -
                              (1.2 + 0.4 - 0.1) * 0.6 + 0.8)
    expected_constraint_value = (0.3 - (-0.2 / 2.3 + 0.7 * -0.5) -
                                 (1.2 + 2.3 - 0.1) * 0.6 + 0.8)

    with self.wrapped_session() as session:
      self.assertNear(
          expected_penalty_value,
          session.run(actual_penalty_value(structure_memoizer)),
          err=1e-6)
      self.assertNear(
          expected_constraint_value,
          session.run(actual_constraint_value(structure_memoizer)),
          err=1e-6)

  def test_variables(self):
    """Tests that `Expression`s propagate extra variables correctly."""
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    def create_dummy_expression(penalty_variable, constraint_variable):
      """Creates an empty `Expression` from the given extra variables."""
      return expression.ExplicitExpression(
          basic_expression.BasicExpression([term.TensorTerm(penalty_variable)]),
          basic_expression.BasicExpression(
              [term.TensorTerm(constraint_variable)]))

    penalty_variable1 = deferred_tensor.DeferredVariable(1)
    penalty_variable2 = deferred_tensor.DeferredVariable(2)
    penalty_variable3 = deferred_tensor.DeferredVariable(3)

    constraint_variable1 = deferred_tensor.DeferredVariable(-1)
    constraint_variable2 = deferred_tensor.DeferredVariable(-2)
    constraint_variable3 = deferred_tensor.DeferredVariable(-3)

    expression1 = create_dummy_expression(penalty_variable1,
                                          constraint_variable1)
    expression2 = create_dummy_expression(penalty_variable2,
                                          constraint_variable2)
    expression3 = create_dummy_expression(penalty_variable3,
                                          constraint_variable3)

    expression12 = expression1 * 0.5 + expression2
    expression23 = expression2 - expression3 / 1.3
    expression123 = -expression12 + 0.6 * expression23

    expression12_penalty_value = (
        expression12.penalty_expression.evaluate(structure_memoizer))
    expression23_penalty_value = (
        expression23.penalty_expression.evaluate(structure_memoizer))
    expression123_penalty_value = (
        expression123.penalty_expression.evaluate(structure_memoizer))

    expression12_constraint_value = (
        expression12.constraint_expression.evaluate(structure_memoizer))
    expression23_constraint_value = (
        expression23.constraint_expression.evaluate(structure_memoizer))
    expression123_constraint_value = (
        expression123.constraint_expression.evaluate(structure_memoizer))

    self.assertEqual(expression12_penalty_value.variables,
                     [penalty_variable1, penalty_variable2])
    self.assertEqual(expression23_penalty_value.variables,
                     [penalty_variable2, penalty_variable3])
    self.assertEqual(expression123_penalty_value.variables,
                     [penalty_variable1, penalty_variable2, penalty_variable3])

    self.assertEqual(expression12_constraint_value.variables,
                     [constraint_variable1, constraint_variable2])
    self.assertEqual(expression23_constraint_value.variables,
                     [constraint_variable2, constraint_variable3])
    self.assertEqual(
        expression123_constraint_value.variables,
        [constraint_variable1, constraint_variable2, constraint_variable3])

  def test_extra_constraints(self):
    """Tests that `Expression`s propagate extra constraints correctly."""

    def create_dummy_expression(extra_constraints=None):
      """Creates an empty `Expression` with the given extra constraints."""
      return expression.ConstrainedExpression(
          expression.ExplicitExpression(
              basic_expression.BasicExpression([]),
              basic_expression.BasicExpression([])),
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

  def test_sum_expression(self):
    """Tests that `SumExpression`s flatten correctly."""
    # The logic of SumExpression is checked in the above tests (which include
    # addition and subtraction). Here, we only check that constructing a
    # SumExpression flattens the list.
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    term_values = [0, 1, 2, 3, 4]

    def create_dummy_expression(value):
      """Creates an empty `Expression` with the given extra constraints."""
      basic_expression_object = basic_expression.BasicExpression(
          [term.TensorTerm(value)])
      return expression.ExplicitExpression(basic_expression_object,
                                           basic_expression_object)

    expressions = [create_dummy_expression(value) for value in term_values]

    # Each of our Expressions contains exactly one term, so by checking its
    # value we can uniquely determine which subexpression is which.
    def term_value(expression_object):
      terms = expression_object.penalty_expression._terms
      self.assertEqual(1, len(terms))
      return terms[0].tensor(structure_memoizer)

    sum1 = expression.SumExpression([expressions[0], expressions[1]])
    sum2 = expression.SumExpression([expressions[2]])
    sum3 = expression.SumExpression([expressions[3]])
    sum4 = expression.SumExpression([expressions[4]])
    sum5 = expression.SumExpression([sum3, sum4])
    sum6 = expression.SumExpression([sum1, sum2, sum5])

    actual_expressions = sum6._expressions
    self.assertEqual(5, len(actual_expressions))
    for ii in xrange(5):
      self.assertEqual(ii, term_value(expressions[ii]))
      self.assertEqual(ii, term_value(actual_expressions[ii]))

  def test_bounded_expression(self):
    """Tests that `BoundedExpression`s select their components correctly."""
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    term1 = term.TensorTerm(1.0)
    term2 = term.TensorTerm(2.0)
    term3 = term.TensorTerm(4.0)
    term4 = term.TensorTerm(8.0)

    basic_expression1 = basic_expression.BasicExpression([term1])
    basic_expression2 = basic_expression.BasicExpression([term2])
    basic_expression3 = basic_expression.BasicExpression([term3])
    basic_expression4 = basic_expression.BasicExpression([term4])

    expression1 = expression.ExplicitExpression(basic_expression1,
                                                basic_expression1)
    expression2 = expression.ExplicitExpression(basic_expression2,
                                                basic_expression2)
    expression3 = expression.ExplicitExpression(basic_expression3,
                                                basic_expression3)
    expression4 = expression.ExplicitExpression(basic_expression4,
                                                basic_expression4)

    # Each of our BasicExpressions contains exactly one term, and while we might
    # negate it, by taking the absolute value we can uniquely determine which
    # BasicExpression is which.
    def term_value(expression_object):
      terms = expression_object.penalty_expression._terms
      self.assertEqual(1, len(terms))
      return abs(terms[0].tensor(structure_memoizer))

    bounded_expression1 = expression.BoundedExpression(
        lower_bound=expression1, upper_bound=expression2)
    self.assertEqual(term_value(bounded_expression1), 2.0)
    self.assertEqual(term_value(-bounded_expression1), 1.0)

    bounded_expression2 = expression.BoundedExpression(
        lower_bound=expression3, upper_bound=expression4)
    self.assertEqual(term_value(bounded_expression2), 8.0)
    self.assertEqual(term_value(-bounded_expression2), 4.0)

    bounded_expression3 = -(bounded_expression1 - bounded_expression2)
    self.assertEqual(term_value(bounded_expression3), 8.0 - 1.0)
    self.assertEqual(term_value(-bounded_expression3), 4.0 - 2.0)

    # Checks that nested BoundedExpressions work.
    bounded_expression4 = expression.BoundedExpression(
        lower_bound=bounded_expression1, upper_bound=expression3)
    self.assertEqual(term_value(bounded_expression4), 4.0)
    self.assertEqual(term_value(-bounded_expression4), 1.0)

    # Checks that nested negated BoundedExpressions work.
    bounded_expression5 = expression.BoundedExpression(
        lower_bound=-bounded_expression1, upper_bound=-bounded_expression2)
    self.assertEqual(term_value(bounded_expression5), 4.0)
    self.assertEqual(term_value(-bounded_expression5), 2.0)

  def test_error_expression(self):
    """Tests that `InvalidExpression`s raise when used."""
    error_expression = expression.InvalidExpression("an error message")
    # All three of "penalty_expression", "constraint_expression" and
    # "extra_constraints" should raise.
    with self.assertRaises(RuntimeError):
      _ = error_expression.penalty_expression
    with self.assertRaises(RuntimeError):
      _ = error_expression.constraint_expression
    with self.assertRaises(RuntimeError):
      _ = error_expression.extra_constraints


if __name__ == "__main__":
  tf.test.main()
