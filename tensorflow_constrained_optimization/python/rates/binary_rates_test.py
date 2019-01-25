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
"""Tests for binary_rates.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import binary_rates
from tensorflow_constrained_optimization.python.rates import subsettable_context


class RatesTest(tf.test.TestCase):
  """Tests for rate-constructing functions."""

  def __init__(self, *args, **kwargs):
    super(RatesTest, self).__init__(*args, **kwargs)

    # We use a fixed fake dataset to make sure that the tests are reproducible.
    # The code for generating this random dataset is:
    #
    #   self._penalty_predictions = np.random.randn(penalty_size)
    #   self._penalty_labels = (
    #       np.random.randint(0, 2, size=penalty_size) * 2 - 1)
    #   self._penalty_weights = np.random.rand(penalty_size)
    #   self._penalty_predicate = np.random.choice(
    #       [False, True], size=penalty_size)
    #
    #   self._constraint_predictions = np.random.randn(constraint_size)
    #   self._constraint_labels = (
    #       np.random.randint(0, 2, size=constraint_size) * 2 - 1)
    #   self._constraint_weights = np.random.rand(constraint_size)
    #   self._constraint_predicate = np.random.choice(
    #       [False, True], size=constraint_size)
    #
    # The dataset itself is:
    self._penalty_predictions = np.array([
        -0.672352809534, 0.814787452952, -0.589617508138, 0.476143711622,
        -0.914392995804, -0.140519198247, 1.01713287656, -0.842355386349,
        -1.86878605935, 0.0312541446545, 0.0929898206701, 1.02580838489
    ])
    self._penalty_labels = np.array([1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1])
    self._penalty_weights = np.array([
        0.305792297057, 0.432750439411, 0.713731859892, 0.635314810893,
        0.513210439781, 0.739280862773, 0.349403785117, 0.0256588613944,
        0.96953248529, 0.257894870733, 0.106252982739, 0.0707090045685
    ])
    self._penalty_predicate = np.array([
        False, False, True, True, False, True, True, False, False, False, True,
        False
    ])
    self._constraint_predictions = np.array([
        -0.942162204902, -0.170590351989, 0.407191043958, 0.224967036781,
        -0.319641536386, 0.0965997062235, 1.58882795293, 0.954582543677
    ])
    self._constraint_labels = np.array([-1, -1, 1, 1, 1, -1, 1, 1])
    self._constraint_weights = np.array([
        0.230071692522, 0.846932765888, 0.265366126241, 0.779471652816,
        0.517297199127, 0.178307346815, 0.354665564039, 0.937435720249
    ])
    self._constraint_predicate = np.array(
        [True, False, False, True, True, True, True, False])

  @property
  def context(self):
    # We can't create the context in __init__, since it would then wind up in
    # the wrong TensorFlow graph.
    context = subsettable_context.split_rate_context(
        penalty_predictions=tf.constant(
            self._penalty_predictions, dtype=tf.float32),
        constraint_predictions=tf.constant(
            self._constraint_predictions, dtype=tf.float32),
        penalty_labels=tf.constant(self._penalty_labels, dtype=tf.float32),
        constraint_labels=tf.constant(
            self._constraint_labels, dtype=tf.float32),
        penalty_weights=tf.constant(self._penalty_weights, dtype=tf.float32),
        constraint_weights=tf.constant(
            self._constraint_weights, dtype=tf.float32))
    return context.subset(self._penalty_predicate, self._constraint_predicate)

  def check_rates(self, expected_penalty_value, expected_constraint_value,
                  actual_expression):
    denominator_lower_bound = 0.0
    global_step = tf.Variable(0, dtype=tf.int32)
    evaluation_context = basic_expression.BasicExpression.EvaluationContext(
        denominator_lower_bound, global_step)

    actual_penalty_value, penalty_pre_train_ops, _ = (
        actual_expression.penalty_expression.evaluate(evaluation_context))
    actual_constraint_value, constraint_pre_train_ops, _ = (
        actual_expression.constraint_expression.evaluate(evaluation_context))

    with self.session() as session:
      session.run(
          [tf.global_variables_initializer(),
           tf.local_variables_initializer()])

      # We only need to run the pre-train ops once, since the entire dataset is
      # contained within the Tensors, so the denominators will be correct.
      session.run(list(penalty_pre_train_ops | constraint_pre_train_ops))

      self.assertAllClose(
          expected_penalty_value,
          session.run(actual_penalty_value),
          rtol=0,
          atol=1e-6)
      self.assertAllClose(
          expected_constraint_value,
          session.run(actual_constraint_value),
          rtol=0,
          atol=1e-6)

  def test_positive_prediction_rate(self):
    """Checks that `positive_prediction_rate` calculates the right quantity."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 + self._penalty_predictions) * self._penalty_weights
        * self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.positive_prediction_rate(self.context)
    self.check_rates(expected_penalty_value, expected_constraint_value,
                     actual_expression)

  def test_negative_prediction_rate(self):
    """Checks that `negative_prediction_rate` calculates the right quantity."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 - self._penalty_predictions) * self._penalty_weights
        * self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 - np.sign(self._constraint_predictions))) *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.negative_prediction_rate(self.context)
    self.check_rates(expected_penalty_value, expected_constraint_value,
                     actual_expression)

  def test_error_rate(self):
    """Checks that `error_rate` calculates the right quantity."""
    # For the penalty, the default loss is hinge.
    expected_signed_penalty_labels = (self._penalty_labels > 0.0) * 2.0 - 1.0
    expected_penalty_numerator = np.sum(
        np.maximum(
            0.0,
            1.0 - expected_signed_penalty_labels * self._penalty_predictions) *
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_signed_constraint_labels = (
        (self._constraint_labels > 0.0) * 2.0 - 1.0)
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 - expected_signed_constraint_labels * np.sign(
            self._constraint_predictions))) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.error_rate(self.context)
    self.check_rates(expected_penalty_value, expected_constraint_value,
                     actual_expression)

  def test_accuracy_rate(self):
    """Checks that `accuracy_rate` calculates the right quantity."""
    # For the penalty, the default loss is hinge.
    expected_signed_penalty_labels = (self._penalty_labels > 0.0) * 2.0 - 1.0
    expected_penalty_numerator = np.sum(
        np.maximum(
            0.0,
            1.0 + expected_signed_penalty_labels * self._penalty_predictions) *
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_signed_constraint_labels = (
        (self._constraint_labels > 0.0) * 2.0 - 1.0)
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 + expected_signed_constraint_labels * np.sign(
            self._constraint_predictions))) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.accuracy_rate(self.context)
    self.check_rates(expected_penalty_value, expected_constraint_value,
                     actual_expression)

  def test_true_positive_rate(self):
    """Checks that `true_positive_rate` calculates the right quantity."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(
            0.0, 1.0 + self._penalty_predictions) * (self._penalty_labels > 0.0)
        * self._penalty_weights * self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        (self._penalty_labels > 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
        (self._constraint_labels > 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(
        (self._constraint_labels > 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.true_positive_rate(self.context)
    self.check_rates(expected_penalty_value, expected_constraint_value,
                     actual_expression)

  def test_false_negative_rate(self):
    """Checks that `false_negative_rate` calculates the right quantity."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(
            0.0, 1.0 - self._penalty_predictions) * (self._penalty_labels > 0.0)
        * self._penalty_weights * self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        (self._penalty_labels > 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 - np.sign(self._constraint_predictions))) *
        (self._constraint_labels > 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(
        (self._constraint_labels > 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.false_negative_rate(self.context)
    self.check_rates(expected_penalty_value, expected_constraint_value,
                     actual_expression)

  def test_false_positive_rate(self):
    """Checks that `false_positive_rate` calculates the right quantity."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 + self._penalty_predictions) *
        (self._penalty_labels <= 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        (self._penalty_labels <= 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
        (self._constraint_labels <= 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(
        (self._constraint_labels <= 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.false_positive_rate(self.context)
    self.check_rates(expected_penalty_value, expected_constraint_value,
                     actual_expression)

  def test_true_negative_rate(self):
    """Checks that `true_negative_rate` calculates the right quantity."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 - self._penalty_predictions) *
        (self._penalty_labels <= 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        (self._penalty_labels <= 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 - np.sign(self._constraint_predictions))) *
        (self._constraint_labels <= 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(
        (self._constraint_labels <= 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.true_negative_rate(self.context)
    self.check_rates(expected_penalty_value, expected_constraint_value,
                     actual_expression)


if __name__ == "__main__":
  tf.test.main()
