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
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import binary_rates
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import subsettable_context
from tensorflow_constrained_optimization.python.rates import test_helpers
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class BinaryRatesTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for binary classification rate-constructing functions."""

  def __init__(self, *args, **kwargs):
    super(BinaryRatesTest, self).__init__(*args, **kwargs)

    self._penalty_size = 12
    self._constraint_size = 8

    # We use a fixed fake dataset to make sure that the tests are reproducible.
    # The code for generating this random dataset is:
    #
    #   self._penalty_predictions = np.random.randn(self._penalty_size)
    #   self._penalty_labels = (
    #       np.random.randint(0, 2, size=self._penalty_size) * 2 - 1)
    #   self._penalty_weights = np.random.rand(self._penalty_size)
    #   self._penalty_predicate = np.random.choice(
    #       [False, True], size=self._penalty_size)
    #
    #   self._constraint_predictions = np.random.randn(self._constraint_size)
    #   self._constraint_labels = (
    #       np.random.randint(0, 2, size=self._constraint_size) * 2 - 1)
    #   self._constraint_weights = np.random.rand(self._constraint_size)
    #   self._constraint_predicate = np.random.choice(
    #       [False, True], size=self._constraint_size)
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
  def _context(self):
    """Creates a new non-split and non-subsetted context."""
    # We can't create the context in __init__, since it would then wind up in
    # the wrong TensorFlow graph.
    predictions = tf.constant(self._penalty_predictions, dtype=tf.float32)
    labels = tf.constant(self._penalty_labels, dtype=tf.float32)
    weights = tf.constant(self._penalty_weights, dtype=tf.float32)
    return subsettable_context.rate_context(
        predictions=lambda: predictions,
        labels=lambda: labels,
        weights=lambda: weights)

  @property
  def _split_context(self):
    """Creates a new split and subsetted context."""
    # We can't create the context in __init__, since it would then wind up in
    # the wrong TensorFlow graph.
    penalty_predictions = tf.constant(
        self._penalty_predictions, dtype=tf.float32)
    constraint_predictions = tf.constant(
        self._constraint_predictions, dtype=tf.float32)
    penalty_labels = tf.constant(self._penalty_labels, dtype=tf.float32)
    constraint_labels = tf.constant(self._constraint_labels, dtype=tf.float32)
    penalty_weights = tf.constant(self._penalty_weights, dtype=tf.float32)
    constraint_weights = tf.constant(self._constraint_weights, dtype=tf.float32)
    context = subsettable_context.split_rate_context(
        penalty_predictions=lambda: penalty_predictions,
        constraint_predictions=lambda: constraint_predictions,
        penalty_labels=lambda: penalty_labels,
        constraint_labels=lambda: constraint_labels,
        penalty_weights=lambda: penalty_weights,
        constraint_weights=lambda: constraint_weights)
    return context.subset(self._penalty_predicate, self._constraint_predicate)

  # FUTURE WORK: this is identical to the corresponding function in
  # multiclass_rates_test.py. Maybe put this in some common place?
  def _check_rates(self, expected_penalty_value, expected_constraint_value,
                   actual_expression):
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    actual_penalty_value = actual_expression.penalty_expression.evaluate(
        structure_memoizer)
    actual_constraint_value = actual_expression.constraint_expression.evaluate(
        structure_memoizer)

    # We need to explicitly create the variables before creating the wrapped
    # session.
    variables = deferred_tensor.DeferredVariableList(
        actual_penalty_value.variables + actual_constraint_value.variables).list
    for variable in variables:
      variable.create(structure_memoizer)

    def update_ops_fn():
      update_ops = []
      for variable in variables:
        update_ops += variable.update_ops(structure_memoizer)
      return update_ops

    with self.wrapped_session() as session:
      # We only need to run the update ops once, since the entire dataset is
      # contained within the Tensors, so the denominators will be correct.
      session.run_ops(update_ops_fn)

      self.assertAllClose(
          expected_penalty_value,
          session.run(actual_penalty_value(structure_memoizer)),
          rtol=0,
          atol=1e-6)
      self.assertAllClose(
          expected_constraint_value,
          session.run(actual_constraint_value(structure_memoizer)),
          rtol=0,
          atol=1e-6)

  def test_positive_prediction_rate(self):
    """Checks `positive_prediction_rate`."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 + self._penalty_predictions) *
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.positive_prediction_rate(
        self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_negative_prediction_rate(self):
    """Checks `negative_prediction_rate`."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 - self._penalty_predictions) *
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 - np.sign(self._constraint_predictions))) *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.negative_prediction_rate(
        self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_error_rate(self):
    """Checks `error_rate`."""
    # For the penalty, the default loss is hinge.
    expected_signed_penalty_labels = (self._penalty_labels > 0.0) * 2.0 - 1.0
    expected_penalty_numerator = np.sum(
        np.maximum(
            0.0,
            1.0 - expected_signed_penalty_labels * self._penalty_predictions) *
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_signed_constraint_labels = ((self._constraint_labels > 0.0) * 2.0 -
                                         1.0)
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 - expected_signed_constraint_labels *
                np.sign(self._constraint_predictions))) *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.error_rate(self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_accuracy_rate(self):
    """Checks `accuracy_rate`."""
    # For the penalty, the default loss is hinge.
    expected_signed_penalty_labels = (self._penalty_labels > 0.0) * 2.0 - 1.0
    expected_penalty_numerator = np.sum(
        np.maximum(
            0.0,
            1.0 + expected_signed_penalty_labels * self._penalty_predictions) *
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_signed_constraint_labels = ((self._constraint_labels > 0.0) * 2.0 -
                                         1.0)
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 + expected_signed_constraint_labels *
                np.sign(self._constraint_predictions))) *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.accuracy_rate(self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_true_positive_rate(self):
    """Checks `true_positive_rate`."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 + self._penalty_predictions) *
        (self._penalty_labels > 0.0) * self._penalty_weights *
        self._penalty_predicate)
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

    actual_expression = binary_rates.true_positive_rate(self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_false_negative_rate(self):
    """Checks `false_negative_rate`."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 - self._penalty_predictions) *
        (self._penalty_labels > 0.0) * self._penalty_weights *
        self._penalty_predicate)
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

    actual_expression = binary_rates.false_negative_rate(self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_false_positive_rate(self):
    """Checks `false_positive_rate`."""
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

    actual_expression = binary_rates.false_positive_rate(self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_true_negative_rate(self):
    """Checks `true_negative_rate`."""
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

    actual_expression = binary_rates.true_negative_rate(self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_true_positive_proportion(self):
    """Checks `true_positive_proportion`."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 + self._penalty_predictions) *
        (self._penalty_labels > 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
        (self._constraint_labels > 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.true_positive_proportion(
        self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_false_negative_proportion(self):
    """Checks `false_negative_proportion`."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 - self._penalty_predictions) *
        (self._penalty_labels > 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 - np.sign(self._constraint_predictions))) *
        (self._constraint_labels > 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.false_negative_proportion(
        self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_false_positive_proportion(self):
    """Checks `false_positive_proportion`."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 + self._penalty_predictions) *
        (self._penalty_labels <= 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
        (self._constraint_labels <= 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.false_positive_proportion(
        self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_true_negative_proportion(self):
    """Checks `true_negative_proportion`."""
    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 - self._penalty_predictions) *
        (self._penalty_labels <= 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 - np.sign(self._constraint_predictions))) *
        (self._constraint_labels <= 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = binary_rates.true_negative_proportion(
        self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_precision_ratio(self):
    """Checks `precision_ratio`."""
    actual_numerator_expression, actual_denominator_expression = (
        binary_rates.precision_ratio(self._split_context))

    # First check the numerator of the precision (which is a rate that itself
    # has a numerator and denominator).

    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 + self._penalty_predictions) *
        (self._penalty_labels > 0.0) * self._penalty_weights *
        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
        (self._constraint_labels > 0.0) * self._constraint_weights *
        self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_numerator_expression)

    # Next check the denominator of the precision (which is a rate that itself
    # has a numerator and denominator, although this "inner" denominator is the
    # same as above).

    # For the penalty, the default loss is hinge.
    expected_penalty_numerator = np.sum(
        np.maximum(0.0, 1.0 + self._penalty_predictions) *
        self._penalty_weights * self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    # For the constraint, the default loss is zero-one.
    expected_constraint_numerator = np.sum(
        (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_denominator_expression)

  def test_f_score_ratio(self):
    """Checks `f_score_ratio`."""
    # We check the most common choices for the beta parameter to the F-score.
    for beta in [0.0, 0.5, 1.0, 2.0]:
      actual_numerator_expression, actual_denominator_expression = (
          binary_rates.f_score_ratio(self._split_context, beta))

      # First check the numerator of the F-score (which is a rate that itself
      # has a numerator and denominator).

      # For the penalty, the default loss is hinge.
      expected_penalty_numerator = (1.0 + beta * beta) * np.sum(
          np.maximum(0.0, 1.0 + self._penalty_predictions) *
          (self._penalty_labels > 0.0) * self._penalty_weights *
          self._penalty_predicate)
      expected_penalty_denominator = np.sum(self._penalty_weights *
                                            self._penalty_predicate)
      expected_penalty_value = (
          expected_penalty_numerator / expected_penalty_denominator)

      # For the constraint, the default loss is zero-one.
      expected_constraint_numerator = (1.0 + beta * beta) * np.sum(
          (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
          (self._constraint_labels > 0.0) * self._constraint_weights *
          self._constraint_predicate)
      expected_constraint_denominator = np.sum(self._constraint_weights *
                                               self._constraint_predicate)
      expected_constraint_value = (
          expected_constraint_numerator / expected_constraint_denominator)

      self._check_rates(expected_penalty_value, expected_constraint_value,
                        actual_numerator_expression)

      # Next check the denominator of the F-score (which is a rate that itself
      # has a numerator and denominator, although this "inner" denominator is
      # the same as above).

      # For the penalty, the default loss is hinge. Notice that, on the
      # positively-labeled examples, we have positive predictions weighted as
      # (1 + beta^2), and negative predictions weighted as beta^2. Internally,
      # the rate-handling code simplifies this to a beta^2 weight on *all*
      # positively-labeled examples (independently of the model, so there is no
      # hinge loss), plus a weight of 1 on true positives. The idea here is that
      # since what we actually want to constrain are rates, we only bound the
      # quantities that need to be bounded--constants remain as constants.
      # Hence, we do this:
      expected_penalty_numerator = np.sum(
          np.maximum(0.0, 1.0 + self._penalty_predictions) *
          (self._penalty_labels > 0.0) * self._penalty_weights *
          self._penalty_predicate)
      expected_penalty_numerator += (beta * beta) * np.sum(
          (self._penalty_labels > 0.0) * self._penalty_weights *
          self._penalty_predicate)
      # instead of this:
      #   expected_penalty_numerator = (1.0 + beta * beta) * np.sum(
      #       np.maximum(0.0, 1.0 + self._penalty_predictions) *
      #       (self._penalty_labels > 0.0) * self._penalty_weights *
      #       self._penalty_predicate)
      #   expected_penalty_numerator += (beta * beta) * np.sum(
      #       np.maximum(0.0, 1.0 - self._penalty_predictions) *
      #       (self._penalty_labels > 0.0) * self._penalty_weights *
      #       self._penalty_predicate)
      # There is no such issue for the negatively-labeled examples.
      expected_penalty_numerator += np.sum(
          np.maximum(0.0, 1.0 + self._penalty_predictions) *
          (self._penalty_labels <= 0.0) * self._penalty_weights *
          self._penalty_predicate)
      expected_penalty_value = (
          expected_penalty_numerator / expected_penalty_denominator)

      # For the constraint, the default loss is zero-one.
      expected_constraint_numerator += (beta * beta) * np.sum(
          (0.5 * (1.0 - np.sign(self._constraint_predictions))) *
          (self._constraint_labels > 0.0) * self._constraint_weights *
          self._constraint_predicate)
      expected_constraint_numerator += np.sum(
          (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
          (self._constraint_labels <= 0.0) * self._constraint_weights *
          self._constraint_predicate)
      expected_constraint_value = (
          expected_constraint_numerator / expected_constraint_denominator)

      self._check_rates(expected_penalty_value, expected_constraint_value,
                        actual_denominator_expression)

  def _find_roc_auc_thresholds(self, bins):
    """Finds the thresholds associated with each of the ROC AUC bins."""
    indices = [
        index for index in range(len(self._penalty_labels))
        if self._penalty_labels[index] <= 0
    ]
    permutation = sorted(
        indices, key=lambda index: self._penalty_predictions[index])
    denominator = sum(self._penalty_weights[index] for index in permutation)

    # Construct a dictionary mapping thresholds to FPRs.
    fprs = {-float("Inf"): 1.0}
    fpr_numerator = denominator
    for index in permutation:
      fpr_numerator -= self._penalty_weights[index]
      fprs[self._penalty_predictions[index]] = fpr_numerator / denominator

    # These FPR thresholds are the same as in roc_auc_{lower,upper}_bound.
    fpr_thresholds = [(index + 0.5) / bins for index in xrange(bins)]

    # For each FPR threshold, find the threshold on the model output that
    # achieves the desired FPR.
    prediction_thresholds = []
    for fpr_threshold in fpr_thresholds:
      prediction_threshold = min(
          [float("Inf")] +
          [key for key, value in six.iteritems(fprs) if value < fpr_threshold])
      prediction_thresholds.append(prediction_threshold)

    return prediction_thresholds

  def test_roc_auc(self):
    """Tests that roc_auc's constraints give correct thresholds."""
    # We don't check roc_auc_upper_bound since most of the code is shared, and
    # the test is too slow already.
    bins = 3
    bisection_epsilon = 1e-6
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    expression = binary_rates.roc_auc(self._context, bins)

    # Extract the the constraints and the associated variables.
    constraint_list = []
    variables = deferred_tensor.DeferredVariableList()
    for constraint in expression.extra_constraints:
      constraint_value = constraint.expression.constraint_expression.evaluate(
          structure_memoizer)
      constraint_list.append(constraint_value)
      variables += constraint_value.variables
    variables = variables.list
    self.assertEqual(bins, len(constraint_list))
    constraints = deferred_tensor.DeferredTensor.apply(
        lambda *args: tf.stack(args), *constraint_list)

    # We need to explicitly create all variables included in the expression
    # before we can try to extract the roc_auc_thresholds.
    for variable in variables:
      variable.create(structure_memoizer)

    # The find_zeros_of_functions() helper will perform a bisection search over
    # the roc_auc_thresholds, so we need to extract the Tensor containing them
    # from the graph.
    roc_auc_thresholds = None
    for variable in variables:
      tensor = variable(structure_memoizer)
      if tensor.name.startswith("tfco_roc_auc_thresholds"):
        self.assertIsNone(roc_auc_thresholds)
        roc_auc_thresholds = tensor
    self.assertIsNotNone(roc_auc_thresholds)

    def update_ops_fn():
      update_ops = []
      for variable in variables:
        update_ops += variable.update_ops(structure_memoizer)
      return update_ops

    with self.wrapped_session() as session:
      session.run_ops(update_ops_fn)

      def evaluate_fn(values):
        """Assigns the variables and evaluates the constraints."""
        session.run_ops(lambda: roc_auc_thresholds.assign(values))
        return session.run(constraints(structure_memoizer))

      actual_thresholds = test_helpers.find_zeros_of_functions(
          bins, evaluate_fn, epsilon=bisection_epsilon)

    expected_thresholds = self._find_roc_auc_thresholds(bins)
    self.assertAllClose(
        expected_thresholds, actual_thresholds, rtol=0, atol=bisection_epsilon)


if __name__ == "__main__":
  tf.test.main()
