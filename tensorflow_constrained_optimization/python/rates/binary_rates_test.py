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

import math
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import binary_rates
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import subsettable_context
# Placeholder for internal import.


def associate_functions_with_variables(size, evaluate_fn):
  """Finds a 1:1 correspondence between functions and variables.

  Suppose that we have n functions and n variables, and that each of the former
  is a function of *exactly* one of the latter. This situation commonly results
  from our use for thresholds and slack variables.

  This function will determine which functions depend on which variables.

  Args:
    size: int, the number of functions and variables.
    evaluate_fn: function mapping a numpy array of variable values to another
      numpy array of function values.

  Returns:
    A list of length "size", in which the ith element is the function that uses
    the ith variable. Equivalently, this list can be interpreted as a
    permutation that should be applied to the functions, in order to ensure that
    the ith variable is used by the ith function.

  Raises:
    RuntimeError: if we fail to successfully find the assignment. This is most
      likely to happen if there is not actually a 1:1 correspondence between
      functions and variables.
  """
  # FUTURE WORK: make these into (optional) arguments.
  default_value = 1000
  changed_value = -1000

  assignments = []
  for ii in xrange(size):
    assignments.append(set(xrange(size)))

  default_values = evaluate_fn([default_value] * size)

  num_loops = int(math.ceil(math.log(size, 2)))
  bit = 1
  for ii in xrange(num_loops):
    variable_indices1 = [ii for ii in xrange(size) if ii & bit != 0]
    variable_indices2 = [ii for ii in xrange(size) if ii & bit == 0]

    arguments = np.array([default_value] * size)
    arguments[variable_indices1] = changed_value
    values1 = evaluate_fn(arguments)
    function_indices1 = [
        ii for ii in xrange(size) if default_values[ii] != values1[ii]
    ]

    arguments = np.array([default_value] * size)
    arguments[variable_indices2] = changed_value
    values2 = evaluate_fn(arguments)
    function_indices2 = [
        ii for ii in xrange(size) if default_values[ii] != values2[ii]
    ]

    # The functions that changed when the variables in variable_indices2 were
    # changed cannot be functions of the variables in variable_indices1 (and
    # vice-versa).
    for jj in function_indices2:
      assignments[jj] -= set(variable_indices1)
    for jj in function_indices1:
      assignments[jj] -= set(variable_indices2)

    bit += bit

  for ii in xrange(size):
    if len(assignments[ii]) != 1:
      raise RuntimeError("unable to associate variables with functions")
    assignments[ii] = assignments[ii].pop()

  # We need to invert the permutation contained in "assignments", since we want
  # it to act on the functions, not the variables.
  permutation = [0] * size
  for ii in xrange(size):
    permutation[assignments[ii]] = ii

  return permutation


def find_zeros_of_functions(size, evaluate_fn, epsilon=1e-6):
  """Finds the zeros of a set of functions.

  Suppose that we have n functions and n variables, and that each of the former
  is a function of *exactly* one of the latter (we do *not* need to know which
  function depends on which variable: we'll figure it out). This situation
  commonly results from our use for thresholds and slack variables.

  This function will find a value for each variable that (rougly) causes the
  corresponding function to equal zero.

  Args:
    size: int, the number of functions and variables.
    evaluate_fn: function mapping a numpy array of variable values to another
      numpy array of function values.
    epsilon: float, the desired precision of the returned variable values.

  Returns:
    A numpy array of variable assignments that zero the functions.

  Raises:
    RuntimeError: if we fail to successfully find an assignment between
      variables and functions, or if the search fails to terminate.
  """
  # FUTURE WORK: make these into (optional) arguments.
  maximum_initialization_iterations = 16
  maximum_bisection_iterations = 64

  # Find the association between variables and functions.
  permutation = associate_functions_with_variables(size, evaluate_fn)
  permuted_evaluate_fn = lambda values: evaluate_fn(values)[permutation]

  # Find initial lower/upper bounds for bisection search by growing the interval
  # until the two ends have different signs.
  lower_bounds = np.zeros(size)
  upper_bounds = np.ones(size)
  delta = 1.0
  iterations = 0
  while True:
    lower_values = permuted_evaluate_fn(lower_bounds)
    upper_values = permuted_evaluate_fn(upper_bounds)
    same_signs = ((lower_values > 0) == (upper_values > 0))
    if not any(same_signs):
      break
    lower_bounds[same_signs] -= delta
    upper_bounds[same_signs] += delta
    delta += delta

    iterations += 1
    if iterations > maximum_initialization_iterations:
      raise RuntimeError("initial lower/upper bound search did not terminate.")

  # Perform a bisection search to find the zeros.
  iterations = 0
  while True:
    middle_bounds = 0.5 * (lower_bounds + upper_bounds)
    middle_values = permuted_evaluate_fn(middle_bounds)
    same_signs = ((lower_values > 0) == (middle_values > 0))
    different_signs = ~same_signs
    lower_bounds[same_signs] = middle_bounds[same_signs]
    lower_values[same_signs] = middle_values[same_signs]
    upper_bounds[different_signs] = middle_bounds[different_signs]
    upper_values[different_signs] = middle_values[different_signs]
    if max(upper_bounds - lower_bounds) <= epsilon:
      break

    iterations += 1
    if iterations > maximum_bisection_iterations:
      raise RuntimeError("bisection search did not terminate")

  return 0.5 * (lower_bounds + upper_bounds)


# @run_all_tests_in_graph_and_eager_modes
class RatesTest(graph_and_eager_test_case.GraphAndEagerTestCase):
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

  def _check_rates(self, expected_penalty_value, expected_constraint_value,
                   actual_expression):
    memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.compat.v2.Variable(0, dtype=tf.int32)
    }

    actual_penalty_value = actual_expression.penalty_expression.evaluate(
        memoizer)
    actual_constraint_value = actual_expression.constraint_expression.evaluate(
        memoizer)

    # We need to explicitly create the variables before creating the wrapped
    # session.
    variables = deferred_tensor.DeferredVariableList(
        actual_penalty_value.variables + actual_constraint_value.variables).list
    for variable in variables:
      variable.create(memoizer)

    def update_ops_fn():
      update_ops = []
      for variable in variables:
        update_ops += variable.update_ops(memoizer)
      return update_ops

    with self.wrapped_session() as session:
      # We only need to run the update ops once, since the entire dataset is
      # contained within the Tensors, so the denominators will be correct.
      session.run_ops(update_ops_fn)

      self.assertAllClose(
          expected_penalty_value,
          session.run(actual_penalty_value(memoizer)),
          rtol=0,
          atol=1e-6)
      self.assertAllClose(
          expected_constraint_value,
          session.run(actual_constraint_value(memoizer)),
          rtol=0,
          atol=1e-6)

  def test_positive_prediction_rate(self):
    """Checks that `positive_prediction_rate` calculates the right quantity."""
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
    """Checks that `negative_prediction_rate` calculates the right quantity."""
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
    """Checks that `error_rate` calculates the right quantity."""
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
    """Checks that `accuracy_rate` calculates the right quantity."""
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
    """Checks that `true_positive_rate` calculates the right quantity."""
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
    """Checks that `false_negative_rate` calculates the right quantity."""
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

    actual_expression = binary_rates.false_positive_rate(self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
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

    actual_expression = binary_rates.true_negative_rate(self._split_context)
    self._check_rates(expected_penalty_value, expected_constraint_value,
                      actual_expression)

  def test_true_positive_proportion(self):
    """Checks that `true_positive_proportion` calculates the right quantity."""
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
    """Checks that `false_negative_proportion` calculates the right quantity."""
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
    """Checks that `false_positive_proportion` calculates the right quantity."""
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
    """Checks that `true_negative_proportion` calculates the right quantity."""
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
    """Checks that `precision_ratio` calculates the right quantities."""
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

  def test_precision(self):
    """Checks that `precision` calculates the right quantities."""
    bisection_epsilon = 1e-6
    memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.compat.v2.Variable(0, dtype=tf.int32)
    }

    expression = binary_rates.precision(self._split_context)

    # Extract the the constraints and the associated variables.
    constraint_list = []
    variables = deferred_tensor.DeferredVariableList()
    for constraint in expression.extra_constraints:
      constraint_value = constraint.expression.constraint_expression.evaluate(
          memoizer)
      constraint_list.append(constraint_value)
      variables += constraint_value.variables
    variables = variables.list
    self.assertEqual(2, len(constraint_list))
    constraints = deferred_tensor.DeferredTensor.apply(
        lambda *args: tf.stack(args), *constraint_list)

    # We need to explicitly create all variables included in the expression
    # before we can try to extract the ratio_bounds.
    for variable in variables:
      variable.create(memoizer)

    # The find_zeros_of_functions() helper will perform a bisection search over
    # the ratio_bounds, so we need to extract the Tensor containing them from
    # the graph.
    ratio_bounds = None
    for variable in variables:
      tensor = variable(memoizer)
      if tensor.name.startswith("tfco_ratio_bounds"):
        self.assertIsNone(ratio_bounds)
        ratio_bounds = tensor
    self.assertIsNotNone(ratio_bounds)

    def update_ops_fn():
      update_ops = []
      for variable in variables:
        update_ops += variable.update_ops(memoizer)
      return update_ops

    with self.wrapped_session() as session:
      session.run_ops(update_ops_fn)

      def evaluate_fn(values):
        """Assigns the variables and evaluates the constraints."""
        session.run_ops(lambda: ratio_bounds.assign(values))
        return session.run(constraints(memoizer))

      actual_ratio_bounds = find_zeros_of_functions(
          2, evaluate_fn, epsilon=bisection_epsilon)
      actual_numerator = actual_ratio_bounds[0]
      actual_denominator = actual_ratio_bounds[1]

    expected_numerator = (
        np.sum((0.5 * (1.0 + np.sign(self._constraint_predictions))) *
               (self._constraint_labels > 0.0) * self._constraint_weights *
               self._constraint_predicate) /
        np.sum(self._constraint_weights * self._constraint_predicate))

    expected_denominator = (
        np.sum((0.5 * (1.0 + np.sign(self._constraint_predictions))) *
               self._constraint_weights * self._constraint_predicate) /
        np.sum(self._constraint_weights * self._constraint_predicate))

    self.assertAllClose(
        expected_numerator, actual_numerator, rtol=0, atol=bisection_epsilon)
    self.assertAllClose(
        expected_denominator,
        actual_denominator,
        rtol=0,
        atol=bisection_epsilon)

  def test_f_score_ratio(self):
    """Checks that `f_score_ratio` calculates the right quantities."""
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

  def test_f_score(self):
    """Checks that `f_score` calculates the right quantities."""
    beta = 1.6
    bisection_epsilon = 1e-6
    memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.compat.v2.Variable(0, dtype=tf.int32)
    }

    expression = binary_rates.f_score(self._split_context, beta)

    # Extract the the constraints and the associated variables.
    constraint_list = []
    variables = deferred_tensor.DeferredVariableList()
    for constraint in expression.extra_constraints:
      constraint_value = constraint.expression.constraint_expression.evaluate(
          memoizer)
      constraint_list.append(constraint_value)
      variables += constraint_value.variables
    variables = variables.list
    self.assertEqual(2, len(constraint_list))
    constraints = deferred_tensor.DeferredTensor.apply(
        lambda *args: tf.stack(args), *constraint_list)

    # We need to explicitly create all variables included in the expression
    # before we can try to extract the ratio_bounds.
    for variable in variables:
      variable.create(memoizer)

    # The find_zeros_of_functions() helper will perform a bisection search over
    # the ratio_bounds, so we need to extract the Tensor containing them from
    # the graph.
    ratio_bounds = None
    for variable in variables:
      tensor = variable(memoizer)
      if tensor.name.startswith("tfco_ratio_bounds"):
        self.assertIsNone(ratio_bounds)
        ratio_bounds = tensor
    self.assertIsNotNone(ratio_bounds)

    def update_ops_fn():
      update_ops = []
      for variable in variables:
        update_ops += variable.update_ops(memoizer)
      return update_ops

    with self.wrapped_session() as session:
      session.run_ops(update_ops_fn)

      def evaluate_fn(values):
        """Assigns the variables and evaluates the constraints."""
        session.run_ops(lambda: ratio_bounds.assign(values))
        return session.run(constraints(memoizer))

      actual_ratio_bounds = find_zeros_of_functions(
          2, evaluate_fn, epsilon=bisection_epsilon)
      actual_numerator = actual_ratio_bounds[0]
      actual_denominator = actual_ratio_bounds[1]

    expected_numerator = (
        (1.0 + beta * beta) * np.sum(
            (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
            (self._constraint_labels > 0.0) * self._constraint_weights *
            self._constraint_predicate) /
        np.sum(self._constraint_weights * self._constraint_predicate))

    expected_denominator = (((1.0 + beta * beta) * np.sum(
        (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
        (self._constraint_labels > 0.0) * self._constraint_weights *
        self._constraint_predicate) + (beta * beta) * np.sum(
            (0.5 * (1.0 - np.sign(self._constraint_predictions))) *
            (self._constraint_labels > 0.0) * self._constraint_weights *
            self._constraint_predicate) + np.sum(
                (0.5 * (1.0 + np.sign(self._constraint_predictions))) *
                (self._constraint_labels <= 0.0) * self._constraint_weights *
                self._constraint_predicate)) / np.sum(
                    self._constraint_weights * self._constraint_predicate))

    self.assertAllClose(
        expected_numerator, actual_numerator, rtol=0, atol=bisection_epsilon)
    self.assertAllClose(
        expected_denominator,
        actual_denominator,
        rtol=0,
        atol=bisection_epsilon)

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
    memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.compat.v2.Variable(0, dtype=tf.int32)
    }

    expression = binary_rates.roc_auc(self._context, bins)

    # Extract the the constraints and the associated variables.
    constraint_list = []
    variables = deferred_tensor.DeferredVariableList()
    for constraint in expression.extra_constraints:
      constraint_value = constraint.expression.constraint_expression.evaluate(
          memoizer)
      constraint_list.append(constraint_value)
      variables += constraint_value.variables
    variables = variables.list
    self.assertEqual(bins, len(constraint_list))
    constraints = deferred_tensor.DeferredTensor.apply(
        lambda *args: tf.stack(args), *constraint_list)

    # We need to explicitly create all variables included in the expression
    # before we can try to extract the roc_auc_thresholds.
    for variable in variables:
      variable.create(memoizer)

    # The find_zeros_of_functions() helper will perform a bisection search over
    # the roc_auc_thresholds, so we need to extract the Tensor containing them
    # from the graph.
    roc_auc_thresholds = None
    for variable in variables:
      tensor = variable(memoizer)
      if tensor.name.startswith("tfco_roc_auc_thresholds"):
        self.assertIsNone(roc_auc_thresholds)
        roc_auc_thresholds = tensor
    self.assertIsNotNone(roc_auc_thresholds)

    def update_ops_fn():
      update_ops = []
      for variable in variables:
        update_ops += variable.update_ops(memoizer)
      return update_ops

    with self.wrapped_session() as session:
      session.run_ops(update_ops_fn)

      def evaluate_fn(values):
        """Assigns the variables and evaluates the constraints."""
        session.run_ops(lambda: roc_auc_thresholds.assign(values))
        return session.run(constraints(memoizer))

      actual_thresholds = find_zeros_of_functions(
          bins, evaluate_fn, epsilon=bisection_epsilon)

    expected_thresholds = self._find_roc_auc_thresholds(bins)
    self.assertAllClose(
        expected_thresholds, actual_thresholds, rtol=0, atol=bisection_epsilon)


if __name__ == "__main__":
  tf.test.main()
