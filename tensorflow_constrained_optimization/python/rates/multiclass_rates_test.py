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
"""Tests for multiclass_rates.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import multiclass_rates
from tensorflow_constrained_optimization.python.rates import subsettable_context
# Placeholder for internal import.


def zero_one_loss(weights, predictions):
  shape = np.shape(weights)
  if np.shape(predictions) != shape:
    raise ValueError("weights and predictions must have the same shapes")
  if len(shape) != 2:
    raise ValueError("weights and predictions must be two-dimensional")
  if shape != np.shape(predictions):
    raise ValueError("weights and predictions must be have the same shape")
  num_examples, num_classes = shape

  result = []
  for ii in xrange(num_examples):
    best_score = -float("Inf")
    best_count = 0.0
    for jj in xrange(num_classes):
      if predictions[ii, jj] > best_score:
        best_score = predictions[ii, jj]
        best_count = 1.0
      elif predictions[ii, jj] == best_score:
        best_count += 1.0

    count = 0.0
    for jj in xrange(num_classes):
      if predictions[ii, jj] >= best_score:
        count += weights[ii, jj]

    result.append(count / best_count)

  return np.array(result)


def hinge_loss(weights, predictions):
  shape = np.shape(weights)
  if np.shape(predictions) != shape:
    raise ValueError("weights and predictions must have the same shapes")
  if len(shape) != 2:
    raise ValueError("weights and predictions must be two-dimensional")
  if shape != np.shape(predictions):
    raise ValueError("weights and predictions must be have the same shape")
  num_examples, num_classes = shape

  weights_permutation = np.argsort(weights, axis=1)

  result = []
  for ii in xrange(num_examples):
    total_weighted_hinge = 0.0

    excluded_indices = set()
    included_indices = set(range(num_classes))
    for jj in xrange(num_classes - 1):
      included_indices.remove(weights_permutation[ii, jj])
      excluded_indices.add(weights_permutation[ii, jj])

      included_max = -float("Inf")
      for kk in included_indices:
        included_max = max(included_max, predictions[ii, kk])

      excluded_mean = 0.0
      for kk in excluded_indices:
        excluded_mean += predictions[ii, kk]
      excluded_mean /= len(excluded_indices)

      delta_weight = (
          weights[ii, weights_permutation[ii, jj + 1]] -
          weights[ii, weights_permutation[ii, jj]])
      total_weighted_hinge += delta_weight * max(
          0, 1.0 + included_max - excluded_mean)

    value = weights[ii, weights_permutation[ii, 0]] + total_weighted_hinge

    result.append(value)

  return np.array(result)


# @run_all_tests_in_graph_and_eager_modes
class MulticlassRatesTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for binary classification rate-constructing functions."""

  def __init__(self, *args, **kwargs):
    super(MulticlassRatesTest, self).__init__(*args, **kwargs)

    self._penalty_size = 12
    self._constraint_size = 8
    self._num_classes = 4

    # We use a fixed fake dataset to make sure that the tests are reproducible.
    # The code for generating this random dataset is:
    #
    #   self._penalty_predictions = np.random.randn(
    #       self._penalty_size, self._num_classes)
    #   self._penalty_labels = np.random.randint(
    #       0, self._num_classes, size=self._penalty_size)
    #   self._penalty_weights = np.random.rand(self._penalty_size)
    #   self._penalty_predicate = np.random.choice(
    #       [False, True], size=self._penalty_size)
    #
    #   self._constraint_predictions = np.random.randn(
    #       self._constraint_size, self._num_classes)
    #   self._constraint_labels = np.random.randint(
    #       0, self._num_classes, size=self._constraint_size)
    #   self._constraint_weights = np.random.rand(self._constraint_size)
    #   self._constraint_predicate = np.random.choice(
    #       [False, True], size=self._constraint_size)
    #
    # The dataset itself is:
    self._penalty_predictions = np.array(
        [[-1.5970997, 1.877267, -0.66030723, -0.01463978],
         [-1.87999382, -0.00305018, 2.42472298, 0.05893705],
         [-0.11031741, 0.82471499, 0.9340874, 0.09632045],
         [0.49282407, 0.57338305, 0.40928707, -0.61865314],
         [-0.7886149, 0.94948278, 1.96216129, 0.20474539],
         [0.72704683, -1.6208753, -0.31098981, -2.16005564],
         [-0.67164428, 0.37699518, 1.24978421, -0.87508569],
         [0.67631863, -0.15639794, -0.43874642, 0.43672745],
         [-0.34359654, -1.41637908, -0.36718105, -0.36349423],
         [-0.55319159, -0.08677386, 0.86685222, 1.19394724],
         [0.64423552, 0.13959498, -1.25362601, 0.40450444],
         [-0.84070832, 1.34938865, -0.63288385, 0.07019597]])
    self._penalty_labels = np.array([0, 2, 3, 0, 3, 0, 3, 1, 0, 3, 1, 3])
    self._penalty_weights = np.array([
        0.76566858, 0.61112134, 0.09629605, 0.48397956, 0.43083251, 0.54200695,
        0.91410649, 0.22486834, 0.29674182, 0.62188739, 0.43582355, 0.73587001
    ])
    self._penalty_predicate = np.array([
        False, True, False, False, True, False, True, True, False, False, True,
        True
    ])
    self._constraint_predictions = np.array(
        [[-1.02284955, 2.19879824, 1.01087809, 1.2813714],
         [-0.2746204, -1.1608573, 0.08607241, -1.78127669],
         [-0.96669923, 0.66164043, -0.88072148, 2.3059222],
         [1.84892764, 0.23774778, 1.59575183, -0.55435492],
         [0.79456944, -1.31367073, -0.82844754, 1.05074885],
         [0.36645997, -1.31130601, 1.29792815, 0.52346038],
         [-1.55433413, -1.92272332, -1.26217317, 0.41987784],
         [0.02946888, 0.69755685, -0.22851259, -1.20193645]])
    self._constraint_labels = np.array([0, 3, 1, 3, 2, 2, 0, 3])
    self._constraint_weights = np.array([
        0.90018779, 0.08155366, 0.87932082, 0.45599071, 0.03253726, 0.35871828,
        0.74693516, 0.03862526
    ])
    self._constraint_predicate = np.array(
        [False, True, True, False, True, True, True, True])

  @property
  def _split_context(self):
    """Creates a new split and subsetted context."""
    # We can't create the context in __init__, since it would then wind up in
    # the wrong TensorFlow graph.
    penalty_predictions = tf.constant(
        self._penalty_predictions, dtype=tf.float32)
    constraint_predictions = tf.constant(
        self._constraint_predictions, dtype=tf.float32)
    penalty_labels = tf.one_hot(self._penalty_labels, depth=self._num_classes)
    constraint_labels = tf.one_hot(
        self._constraint_labels, depth=self._num_classes)
    penalty_weights = tf.constant(self._penalty_weights, dtype=tf.float32)
    constraint_weights = tf.constant(self._constraint_weights, dtype=tf.float32)
    context = subsettable_context.multiclass_split_rate_context(
        num_classes=self._num_classes,
        penalty_predictions=lambda: penalty_predictions,
        constraint_predictions=lambda: constraint_predictions,
        penalty_labels=lambda: penalty_labels,
        constraint_labels=lambda: constraint_labels,
        penalty_weights=lambda: penalty_weights,
        constraint_weights=lambda: constraint_weights)
    return context.subset(self._penalty_predicate, self._constraint_predicate)

  # FUTURE WORK: this is identical to the corresponding function in
  # binary_rates_test.py. Maybe put this in some common place?
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
    positive_class = 0

    penalty_weights = np.zeros((self._penalty_size, self._num_classes))
    penalty_weights[:, positive_class] = 1.0
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_weights = np.zeros((self._constraint_size, self._num_classes))
    constraint_weights[:, positive_class] = 1.0
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(constraint_losses *
                                           self._constraint_weights *
                                           self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.positive_prediction_rate(
        self._split_context, positive_class=positive_class)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_negative_prediction_rate(self):
    """Checks `negative_prediction_rate`."""
    positive_class = 2

    penalty_weights = np.ones((self._penalty_size, self._num_classes))
    penalty_weights[:, positive_class] = 0.0
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_weights = np.ones((self._constraint_size, self._num_classes))
    constraint_weights[:, positive_class] = 0.0
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(constraint_losses *
                                           self._constraint_weights *
                                           self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.negative_prediction_rate(
        self._split_context, positive_class=positive_class)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_error_rate(self):
    """Checks `error_rate`."""
    penalty_weights = np.ones((self._penalty_size, self._num_classes))
    for ii in xrange(self._penalty_size):
      penalty_weights[ii, self._penalty_labels[ii]] = 0.0
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_weights = np.ones((self._constraint_size, self._num_classes))
    for ii in xrange(self._constraint_size):
      constraint_weights[ii, self._constraint_labels[ii]] = 0.0
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(constraint_losses *
                                           self._constraint_weights *
                                           self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.error_rate(self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_accuracy_rate(self):
    """Checks `accuracy_rate`."""
    penalty_weights = np.zeros((self._penalty_size, self._num_classes))
    for ii in xrange(self._penalty_size):
      penalty_weights[ii, self._penalty_labels[ii]] = 1.0
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_weights = np.zeros((self._constraint_size, self._num_classes))
    for ii in xrange(self._constraint_size):
      constraint_weights[ii, self._constraint_labels[ii]] = 1.0
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(constraint_losses *
                                           self._constraint_weights *
                                           self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.accuracy_rate(self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_true_positive_rate(self):
    """Checks `true_positive_rate`."""
    positive_class = [True, False, False, True]
    class_weights = np.array(positive_class, dtype=np.float32)

    penalty_label_weights = np.zeros(self._penalty_size)
    for ii in xrange(self._penalty_size):
      if positive_class[self._penalty_labels[ii]]:
        penalty_label_weights[ii] = 1.0
    penalty_weights = np.tensordot(penalty_label_weights, class_weights, axes=0)
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * penalty_label_weights *
                                        self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        penalty_label_weights * self._penalty_weights * self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_label_weights = np.zeros(self._constraint_size)
    for ii in xrange(self._constraint_size):
      if positive_class[self._constraint_labels[ii]]:
        constraint_label_weights[ii] = 1.0
    constraint_weights = np.tensordot(
        constraint_label_weights, class_weights, axes=0)
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(
        constraint_losses * constraint_label_weights *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(constraint_label_weights *
                                             self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.true_positive_rate(
        self._split_context, positive_class)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_false_negative_rate(self):
    """Checks `false_negative_rate`."""
    positive_class = [True, False, True, True]
    class_weights = 1.0 - np.array(positive_class, dtype=np.float32)

    penalty_label_weights = np.zeros(self._penalty_size)
    for ii in xrange(self._penalty_size):
      if positive_class[self._penalty_labels[ii]]:
        penalty_label_weights[ii] = 1.0
    penalty_weights = np.tensordot(penalty_label_weights, class_weights, axes=0)
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * penalty_label_weights *
                                        self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        penalty_label_weights * self._penalty_weights * self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_label_weights = np.zeros(self._constraint_size)
    for ii in xrange(self._constraint_size):
      if positive_class[self._constraint_labels[ii]]:
        constraint_label_weights[ii] = 1.0
    constraint_weights = np.tensordot(
        constraint_label_weights, class_weights, axes=0)
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(
        constraint_losses * constraint_label_weights *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(constraint_label_weights *
                                             self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.false_negative_rate(
        self._split_context, positive_class)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_false_positive_rate(self):
    """Checks `false_positive_rate`."""
    positive_class = [False, False, False, True]
    class_weights = np.array(positive_class, dtype=np.float32)

    penalty_label_weights = np.ones(self._penalty_size)
    for ii in xrange(self._penalty_size):
      if positive_class[self._penalty_labels[ii]]:
        penalty_label_weights[ii] = 0.0
    penalty_weights = np.tensordot(penalty_label_weights, class_weights, axes=0)
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * penalty_label_weights *
                                        self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        penalty_label_weights * self._penalty_weights * self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_label_weights = np.ones(self._constraint_size)
    for ii in xrange(self._constraint_size):
      if positive_class[self._constraint_labels[ii]]:
        constraint_label_weights[ii] = 0.0
    constraint_weights = np.tensordot(
        constraint_label_weights, class_weights, axes=0)
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(
        constraint_losses * constraint_label_weights *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(constraint_label_weights *
                                             self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.false_positive_rate(
        self._split_context, positive_class)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_true_negative_rate(self):
    """Checks `true_negative_rate`."""
    positive_class = [False, False, True, True]
    class_weights = 1.0 - np.array(positive_class, dtype=np.float32)

    penalty_label_weights = np.ones(self._penalty_size)
    for ii in xrange(self._penalty_size):
      if positive_class[self._penalty_labels[ii]]:
        penalty_label_weights[ii] = 0.0
    penalty_weights = np.tensordot(penalty_label_weights, class_weights, axes=0)
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * penalty_label_weights *
                                        self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(
        penalty_label_weights * self._penalty_weights * self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_label_weights = np.ones(self._constraint_size)
    for ii in xrange(self._constraint_size):
      if positive_class[self._constraint_labels[ii]]:
        constraint_label_weights[ii] = 0.0
    constraint_weights = np.tensordot(
        constraint_label_weights, class_weights, axes=0)
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(
        constraint_losses * constraint_label_weights *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(constraint_label_weights *
                                             self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.true_negative_rate(
        self._split_context, positive_class)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_true_positive_proportion(self):
    """Checks `true_positive_proportion`."""
    positive_class = [True, False, False, True]
    class_weights = np.array(positive_class, dtype=np.float32)

    penalty_label_weights = np.zeros(self._penalty_size)
    for ii in xrange(self._penalty_size):
      if positive_class[self._penalty_labels[ii]]:
        penalty_label_weights[ii] = 1.0
    penalty_weights = np.tensordot(penalty_label_weights, class_weights, axes=0)
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * penalty_label_weights *
                                        self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_label_weights = np.zeros(self._constraint_size)
    for ii in xrange(self._constraint_size):
      if positive_class[self._constraint_labels[ii]]:
        constraint_label_weights[ii] = 1.0
    constraint_weights = np.tensordot(
        constraint_label_weights, class_weights, axes=0)
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(
        constraint_losses * constraint_label_weights *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.true_positive_proportion(
        self._split_context, positive_class)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_false_negative_proportion(self):
    """Checks `false_negative_proportion`."""
    positive_class = [True, False, True, True]
    class_weights = 1.0 - np.array(positive_class, dtype=np.float32)

    penalty_label_weights = np.zeros(self._penalty_size)
    for ii in xrange(self._penalty_size):
      if positive_class[self._penalty_labels[ii]]:
        penalty_label_weights[ii] = 1.0
    penalty_weights = np.tensordot(penalty_label_weights, class_weights, axes=0)
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * penalty_label_weights *
                                        self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_label_weights = np.zeros(self._constraint_size)
    for ii in xrange(self._constraint_size):
      if positive_class[self._constraint_labels[ii]]:
        constraint_label_weights[ii] = 1.0
    constraint_weights = np.tensordot(
        constraint_label_weights, class_weights, axes=0)
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(
        constraint_losses * constraint_label_weights *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.false_negative_proportion(
        self._split_context, positive_class)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_false_positive_proportion(self):
    """Checks `false_positive_proportion`."""
    positive_class = [False, False, False, True]
    class_weights = np.array(positive_class, dtype=np.float32)

    penalty_label_weights = np.ones(self._penalty_size)
    for ii in xrange(self._penalty_size):
      if positive_class[self._penalty_labels[ii]]:
        penalty_label_weights[ii] = 0.0
    penalty_weights = np.tensordot(penalty_label_weights, class_weights, axes=0)
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * penalty_label_weights *
                                        self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_label_weights = np.ones(self._constraint_size)
    for ii in xrange(self._constraint_size):
      if positive_class[self._constraint_labels[ii]]:
        constraint_label_weights[ii] = 0.0
    constraint_weights = np.tensordot(
        constraint_label_weights, class_weights, axes=0)
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(
        constraint_losses * constraint_label_weights *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.false_positive_proportion(
        self._split_context, positive_class)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_true_negative_proportion(self):
    """Checks `true_negative_proportion`."""
    positive_class = [False, False, True, True]
    class_weights = 1.0 - np.array(positive_class, dtype=np.float32)

    penalty_label_weights = np.ones(self._penalty_size)
    for ii in xrange(self._penalty_size):
      if positive_class[self._penalty_labels[ii]]:
        penalty_label_weights[ii] = 0.0
    penalty_weights = np.tensordot(penalty_label_weights, class_weights, axes=0)
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * penalty_label_weights *
                                        self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_label_weights = np.ones(self._constraint_size)
    for ii in xrange(self._constraint_size):
      if positive_class[self._constraint_labels[ii]]:
        constraint_label_weights[ii] = 0.0
    constraint_weights = np.tensordot(
        constraint_label_weights, class_weights, axes=0)
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(
        constraint_losses * constraint_label_weights *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    actual_expression = multiclass_rates.true_negative_proportion(
        self._split_context, positive_class)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_precision_ratio(self):
    """Checks `precision_ratio`."""
    positive_class = [False, True, True, False]
    class_weights = np.array(positive_class, dtype=np.float32)

    actual_numerator_expression, actual_denominator_expression = (
        multiclass_rates.precision_ratio(self._split_context, positive_class))

    # First check the numerator of the precision (which is a rate that itself
    # has a numerator and denominator).

    penalty_label_weights = np.zeros(self._penalty_size)
    for ii in xrange(self._penalty_size):
      if positive_class[self._penalty_labels[ii]]:
        penalty_label_weights[ii] = 1.0
    penalty_weights = np.tensordot(penalty_label_weights, class_weights, axes=0)
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * penalty_label_weights *
                                        self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_denominator = np.sum(self._penalty_weights *
                                          self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_label_weights = np.zeros(self._constraint_size)
    for ii in xrange(self._constraint_size):
      if positive_class[self._constraint_labels[ii]]:
        constraint_label_weights[ii] = 1.0
    constraint_weights = np.tensordot(
        constraint_label_weights, class_weights, axes=0)
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(
        constraint_losses * constraint_label_weights *
        self._constraint_weights * self._constraint_predicate)
    expected_constraint_denominator = np.sum(self._constraint_weights *
                                             self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_numerator_expression)

    # Next check the denominator of the precision (which is a rate that itself
    # has a numerator and denominator, although this "inner" denominator is the
    # same as above).

    penalty_weights = np.tile(class_weights, (self._penalty_size, 1))
    # For the penalty, the default loss is hinge.
    penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
    expected_penalty_numerator = np.sum(penalty_losses * self._penalty_weights *
                                        self._penalty_predicate)
    expected_penalty_value = (
        expected_penalty_numerator / expected_penalty_denominator)

    constraint_weights = np.tile(class_weights, (self._constraint_size, 1))
    # For the constraint, the default loss is zero-one.
    constraint_losses = zero_one_loss(constraint_weights,
                                      self._constraint_predictions)
    expected_constraint_numerator = np.sum(constraint_losses *
                                           self._constraint_weights *
                                           self._constraint_predicate)
    expected_constraint_value = (
        expected_constraint_numerator / expected_constraint_denominator)

    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_denominator_expression)

  def test_f_score_ratio(self):
    """Checks `f_score_ratio`."""
    positive_class = [True, False, True, False]
    class_weights = np.array(positive_class, dtype=np.float32)

    # We check the most common choices for the beta parameter to the F-score.
    for beta in [0.0, 0.5, 1.0, 2.0]:
      actual_numerator_expression, actual_denominator_expression = (
          multiclass_rates.f_score_ratio(self._split_context, positive_class,
                                         beta))

      # First check the numerator of the F-score (which is a rate that itself
      # has a numerator and denominator).

      penalty_label_weights = np.zeros(self._penalty_size)
      for ii in xrange(self._penalty_size):
        if positive_class[self._penalty_labels[ii]]:
          penalty_label_weights[ii] = 1.0
      penalty_weights = np.tensordot(
          penalty_label_weights, class_weights, axes=0)
      # For the penalty, the default loss is hinge.
      penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
      expected_penalty_numerator = (1.0 + beta * beta) * np.sum(
          penalty_losses * penalty_label_weights * self._penalty_weights *
          self._penalty_predicate)
      expected_penalty_denominator = np.sum(self._penalty_weights *
                                            self._penalty_predicate)
      expected_penalty_value = (
          expected_penalty_numerator / expected_penalty_denominator)

      constraint_label_weights = np.zeros(self._constraint_size)
      for ii in xrange(self._constraint_size):
        if positive_class[self._constraint_labels[ii]]:
          constraint_label_weights[ii] = 1.0
      constraint_weights = np.tensordot(
          constraint_label_weights, class_weights, axes=0)
      # For the constraint, the default loss is zero-one.
      constraint_losses = zero_one_loss(constraint_weights,
                                        self._constraint_predictions)
      expected_constraint_numerator = (1.0 + beta * beta) * np.sum(
          constraint_losses * constraint_label_weights *
          self._constraint_weights * self._constraint_predicate)
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
      penalty_weights = np.tensordot(
          penalty_label_weights, class_weights, axes=0)
      # For the penalty, the default loss is hinge.
      penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
      expected_penalty_numerator = np.sum(
          penalty_losses * penalty_label_weights * self._penalty_weights *
          self._penalty_predicate)
      expected_penalty_numerator += (beta * beta) * np.sum(
          penalty_label_weights * self._penalty_weights *
          self._penalty_predicate)
      # There is no such issue for the negatively-labeled examples.

      penalty_weights = np.tensordot(
          1.0 - penalty_label_weights, class_weights, axes=0)
      # For the penalty, the default loss is hinge.
      penalty_losses = hinge_loss(penalty_weights, self._penalty_predictions)
      expected_penalty_numerator += np.sum(
          penalty_losses * (1.0 - penalty_label_weights) *
          self._penalty_weights * self._penalty_predicate)

      expected_penalty_value = (
          expected_penalty_numerator / expected_penalty_denominator)

      constraint_weights = np.tensordot(
          constraint_label_weights, class_weights, axes=0)
      # For the constraint, the default loss is zero-one.
      constraint_losses = zero_one_loss(constraint_weights,
                                        self._constraint_predictions)
      expected_constraint_numerator = np.sum(
          constraint_losses * constraint_label_weights *
          self._constraint_weights * self._constraint_predicate)
      expected_constraint_numerator += (beta * beta) * np.sum(
          constraint_label_weights * self._constraint_weights *
          self._constraint_predicate)

      constraint_weights = np.tensordot(
          1.0 - constraint_label_weights, class_weights, axes=0)
      # For the constraint, the default loss is zero-one.
      constraint_losses = zero_one_loss(constraint_weights,
                                        self._constraint_predictions)
      expected_constraint_numerator += np.sum(
          constraint_losses * (1.0 - constraint_label_weights) *
          self._constraint_weights * self._constraint_predicate)

      expected_constraint_value = (
          expected_constraint_numerator / expected_constraint_denominator)

      self._check_rates(expected_penalty_value, expected_constraint_value,
                        actual_denominator_expression)


if __name__ == "__main__":
  tf.test.main()
