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
"""Tests for basic_expression.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import loss
from tensorflow_constrained_optimization.python.rates import predicate
from tensorflow_constrained_optimization.python.rates import term
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class BasicExpressionTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for `BasicExpression` class."""

  def test_merging(self):
    """Checks that `BasicExpression`s merge compatible `Term`s."""
    predictions = deferred_tensor.ExplicitDeferredTensor(
        tf.constant([1.0, -1.0, 0.5], dtype=tf.float32))
    weights1 = deferred_tensor.ExplicitDeferredTensor(1.0)
    weights2 = deferred_tensor.ExplicitDeferredTensor(
        tf.constant([0.7, 0.3, 1.0], dtype=tf.float32))
    numerator_predicate1 = predicate.Predicate(True)
    numerator_predicate2 = predicate.Predicate(
        tf.constant([True, False, False]))
    denominator_predicate1 = predicate.Predicate(True)
    denominator_predicate2 = predicate.Predicate(
        tf.constant([True, False, True]))
    # The two terms have the same predictions and loss, so they're compatible.
    term_object1 = term.BinaryClassificationTerm.ratio(1.0, 0.0, predictions,
                                                       weights1,
                                                       numerator_predicate1,
                                                       denominator_predicate1,
                                                       loss.ZeroOneLoss())
    term_object2 = term.BinaryClassificationTerm.ratio(1.0, 0.0, predictions,
                                                       weights2,
                                                       numerator_predicate2,
                                                       denominator_predicate2,
                                                       loss.ZeroOneLoss())
    self.assertEqual(term_object1.key, term_object2.key)

    expression_object1 = basic_expression.BasicExpression([term_object1])
    self.assertEqual(1, len(expression_object1._terms))
    expression_object2 = basic_expression.BasicExpression([term_object2])
    self.assertEqual(1, len(expression_object2._terms))

    # Check that __init__ correctly merges compatible terms.
    expression_object = basic_expression.BasicExpression(
        [term_object1, term_object2])
    self.assertEqual(1, len(expression_object._terms))
    # Check that __add__ correctly merges compatible terms.
    expression_object = expression_object1 + expression_object2
    self.assertEqual(1, len(expression_object._terms))
    # Check that __sub__ correctly merges compatible terms.
    expression_object = expression_object1 - expression_object2
    self.assertEqual(1, len(expression_object._terms))

  def test_not_merging(self):
    """Checks that `BasicExpression`s don't merge incompatible `Term`s."""
    predictions = deferred_tensor.ExplicitDeferredTensor(
        tf.constant([1.0, -1.0, 0.5], dtype=tf.float32))
    weights1 = deferred_tensor.ExplicitDeferredTensor(1.0)
    weights2 = deferred_tensor.ExplicitDeferredTensor(
        tf.constant([0.7, 0.3, 1.0], dtype=tf.float32))
    numerator_predicate1 = predicate.Predicate(True)
    numerator_predicate2 = predicate.Predicate(
        tf.constant([True, False, False]))
    denominator_predicate1 = predicate.Predicate(True)
    denominator_predicate2 = predicate.Predicate(
        tf.constant([True, False, True]))
    # The two terms have different losses, so they're incompatible.
    term_object1 = term.BinaryClassificationTerm.ratio(1.0, 0.0, predictions,
                                                       weights1,
                                                       numerator_predicate1,
                                                       denominator_predicate1,
                                                       loss.ZeroOneLoss())
    term_object2 = term.BinaryClassificationTerm.ratio(1.0, 0.0, predictions,
                                                       weights2,
                                                       numerator_predicate2,
                                                       denominator_predicate2,
                                                       loss.HingeLoss())
    self.assertNotEqual(term_object1.key, term_object2.key)

    expression_object1 = basic_expression.BasicExpression([term_object1])
    self.assertEqual(1, len(expression_object1._terms))
    expression_object2 = basic_expression.BasicExpression([term_object2])
    self.assertEqual(1, len(expression_object2._terms))

    # Check that __init__ doesn't merge incompatible terms.
    expression_object = basic_expression.BasicExpression(
        [term_object1, term_object2])
    self.assertEqual(2, len(expression_object._terms))
    # Check that __add__ doesn't merge incompatible terms.
    expression_object = expression_object1 + expression_object2
    self.assertEqual(2, len(expression_object._terms))
    # Check that __sub__ doesn't merge incompatible terms.
    expression_object = expression_object1 - expression_object2
    self.assertEqual(2, len(expression_object._terms))

  def test_arithmetic(self):
    """Tests `BasicExpression`'s arithmetic operators."""
    memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.compat.v2.Variable(0, dtype=tf.int32)
    }

    dummy_predictions = deferred_tensor.ExplicitDeferredTensor(
        tf.constant(0, dtype=tf.float32, shape=(1,)))
    dummy_weights = deferred_tensor.ExplicitDeferredTensor(1.0)
    true_predicate = predicate.Predicate(True)

    def ratio_expression(positive_coefficient, negative_coefficient,
                         loss_function):
      term_object = term.BinaryClassificationTerm.ratio(
          positive_coefficient, negative_coefficient, dummy_predictions,
          dummy_weights, true_predicate, true_predicate, loss_function)
      return basic_expression.BasicExpression([term_object])

    def constant_expression(constant):
      return basic_expression.BasicExpression(
          [term.TensorTerm(tf.constant(constant, dtype=tf.float32))])

    # This expression exercises all of the operators. The first and third
    # ratio_expression()s will have the same losses (and everything else except
    # the coefficients), and will therefore be compatible. The second has a
    # different loss, and will be incompatible with the other two.
    expression_object = (
        constant_expression(0.3) -
        (ratio_expression(1.0, 0.0, loss.ZeroOneLoss()) / 2.3 +
         0.7 * ratio_expression(0.5, 0.5, loss.HingeLoss())) +
        (constant_expression(1.2) + ratio_expression(
            0.0, 1.0, loss.ZeroOneLoss()) - constant_expression(0.1)) * 0.6 +
        constant_expression(0.8))

    expected_constant = 0.3 + (1.2 - 0.1) * 0.6 + 0.8
    coefficients = np.array([-1.0 / 2.3, -0.7, 0.6], dtype=np.float32)
    positive_coefficients = np.array([1.0, 0.5, 0.0],
                                     dtype=np.float32) * coefficients
    negative_coefficients = np.array([0.0, 0.5, 1.0],
                                     dtype=np.float32) * coefficients
    # The expected weights for the two zero-one terms will be merged, since
    # they're compatible. There is only one hinge term.
    expected_zero_one_positive_weights = (
        positive_coefficients[0] + positive_coefficients[2])
    expected_zero_one_negative_weights = (
        negative_coefficients[0] + negative_coefficients[2])
    expected_hinge_positive_weights = positive_coefficients[1]
    expected_hinge_negative_weights = negative_coefficients[1]

    # We should have three terms, since the two compatible
    # BinaryClassificationTerms will be merged, and we'll have one TensorTerm.
    expression_terms = expression_object._terms
    expression_binary_classification_terms = [
        tt for tt in expression_terms
        if isinstance(tt, term.BinaryClassificationTerm)
    ]
    expression_tensor_terms = [
        tt for tt in expression_terms if isinstance(tt, term.TensorTerm)
    ]
    self.assertEqual(3, len(expression_terms))
    self.assertEqual(2, len(expression_binary_classification_terms))
    self.assertEqual(1, len(expression_tensor_terms))
    zero_one_term, hinge_term = expression_binary_classification_terms
    if zero_one_term.loss != loss.ZeroOneLoss():
      zero_one_term, hinge_term = hinge_term, zero_one_term
    self.assertEqual(zero_one_term.loss, loss.ZeroOneLoss())
    self.assertEqual(hinge_term.loss, loss.HingeLoss())

    actual_constant = expression_tensor_terms[0].evaluate(memoizer)
    actual_zero_one_positive_weights = (
        zero_one_term.positive_ratio_weights.evaluate(memoizer))
    actual_zero_one_negative_weights = (
        zero_one_term.negative_ratio_weights.evaluate(memoizer))
    actual_hinge_positive_weights = (
        hinge_term.positive_ratio_weights.evaluate(memoizer))
    actual_hinge_negative_weights = (
        hinge_term.negative_ratio_weights.evaluate(memoizer))

    # We need to explicitly create the variables before creating the wrapped
    # session.
    variables = deferred_tensor.DeferredVariableList(
        actual_constant.variables + actual_zero_one_positive_weights.variables +
        actual_zero_one_negative_weights.variables +
        actual_hinge_positive_weights.variables +
        actual_hinge_negative_weights.variables)
    for variable in variables:
      variable.create(memoizer)

    with self.wrapped_session() as session:
      self.assertAllClose(
          expected_constant, actual_constant(memoizer), rtol=0, atol=1e-6)

      self.assertAllClose(
          np.array([expected_zero_one_positive_weights]),
          session.run(actual_zero_one_positive_weights(memoizer)),
          rtol=0,
          atol=1e-6)
      self.assertAllClose(
          np.array([expected_zero_one_negative_weights]),
          session.run(actual_zero_one_negative_weights(memoizer)),
          rtol=0,
          atol=1e-6)
      self.assertAllClose(
          np.array([expected_hinge_positive_weights]),
          session.run(actual_hinge_positive_weights(memoizer)),
          rtol=0,
          atol=1e-6)
      self.assertAllClose(
          np.array([expected_hinge_negative_weights]),
          session.run(actual_hinge_negative_weights(memoizer)),
          rtol=0,
          atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
