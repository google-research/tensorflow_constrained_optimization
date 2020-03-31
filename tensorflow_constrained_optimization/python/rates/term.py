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
"""Contains `Term`s for use in rate constraints.

The purpose of this library is to make it easy to construct and optimize ML
problems for which the goal is to minimize a linear combination of rates,
subject to zero or more constraints on linear combinations of rates. That is:

  minimize:  c11 * rate11 + c12 * rate12 + ... + tensor1
  such that: c21 * rate21 + c22 * rate22 + ... + tensor2 <= 0
             c31 * rate31 + c32 * rate32 + ... + tensor3 <= 0
             ...

Each rate is e.g. an error rate, positive prediction rate, false negative rate
on a certain slice of the data, and so on.

The objective and constraint functions are each represented by an `Expression`,
which captures a linear combination of `Term`s and scalar `Tensor`s. A `Term`
represents the proportion of time that a certain event occurs on a subset of the
data. For example, a `Term` might represent the positive prediction rate on the
set of negatively-labeled examples (the false positive rate). All `Term`s have
an associated loss. This could be the zero-one loss (if we wanted, for example,
the exact false positive rate, as above), or a hinge approximation, or something
else.

The above description is somewhat oversimplified--there are a number of
additional complications. For one, an `Expression` object actually consists of
two `BasicExpression` objects: one for the "penalty" portion of the problem
(which is used when optimizing the model parameters), and the other for the
"constraint" portion (used when optimizing the constraints, e.g. the Lagrange
multipliers, if using the Lagrangian formulation). It is the `BasicExpression`
object that actually represents linear combinations of `Term`s, and it contains
some additional logic for combining "compatible" `Term`s.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy
import numbers
import numpy as np
import six
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import helpers
from tensorflow_constrained_optimization.python.rates import predicate


class _RatioWeights(helpers.RateObject):
  """Object representing a `Tensor` of example weights for a ratio or ratios.

  This is a helper class which is used to find weights for sums of ratios. For
  example, if we want to compute the average positive prediction rate on a
  subset of the data (here, 1{} is an indicator function):
    (mean_i weights[i] 1{i in numerator_subset} 1{predictions[i] > 0})
        / (mean_i weights[i] 1{i in denominator_subset})
  then the "evaluate" method will return a `Tensor` of ratio_weights such that:
    ratio_weights[j] = weights[j] 1{j in numerator_subset}
        / (mean_i weights[i] 1{i in denominator_subset})
  with the consequence that the average positive prediction rate on this subset
  can be written as:
    mean_i ratio_weights[i] 1{predictions[i] > 0}
  such ratio objects can be constructed using the "ratio" class method.

  While the above problem only included one weight per example, a
  `_RatioWeights` object will usually include multiple "columns" of weights. In
  general, the number of columns is the number of possible "outcomes" of the
  classifier, and the weights in each column are those that will be used when
  the corresponding outcome occurs. In binary classification, for example, we'll
  want one column for positive predictions (used when predictions[i] > 0, if
  we're using the zero-one loss), and another for negative predictions (when
  predictions[i] <= 0). For multiclass problems, there will be one column per
  class.

  For a single ratio, the "evaluate" method will return the ratio_weights[]
  `Tensor`, as defined above. However, this class also supports linear
  combinations of ratios, each of which potentially has its own "weights",
  "numerator_subset" and "denominator_subset". To this end, it supports
  arithmetic operations. For example, the sum of two `_RatioWeights`s evaluates
  to the sum of the evaluations of the two `_RatioWeights`s. Subtraction,
  negation, scalar multiplication, and scalar division, are also supported.

  Overall, taking linear combinations into account, the quantities represented
  by this class take the form:
    ratio_weights[j] = sum_k c_k weights_k[j] 1{j in numerator_subset_k}
        / (mean_i weights_k[i] 1{i in denominator_subset_k})
  where c_k is the coefficient of the kth element of the linear combination.
  """

  def __init__(self, ratios, num_columns):
    """Creates a new `_RatioWeights` object.

    A `_RatioWeights` object is a sum of ratios, represented internally as a
    dict mapping denominators to numerators. Each element of this dict
    represents a ratio, and the weights Tensor represented by this object is the
    sum of these ratios. This representation allows us to add ratios with common
    denominators by simply adding their numerators.

    Args:
      ratios: A dict mapping denominator keys to numerators. Each denominator
        key is a (`DeferredTensor`, `Predicate`) pair, the first element being
        the example weights, and the second the predicate indicating which
        examples are included in the denominator. Each numerator is a
        `DeferredTensor` of example weights.
      num_columns: int, the number of "columns" stored in this object. This will
        be the number of possible "outcomes" of the classifier: for binary
          classification, num_columns should be 2, while for multiclass, it
          should be the number of classes.
    """
    self._ratios = ratios
    self._num_columns = num_columns

  @property
  def ratios(self):
    """Returns the sum of ratios represented by this object in a list.

    Returns:
      A list of (denominator_key, numerator) pairs. Each denominator key is a
      (`DeferredTensor`, `Predicate`) pair, the first element being the example
      weights, and the second the predicate indicating which examples are
      included in the denominator. Each numerator is a `DeferredTensor` of
      example weights.
    """
    return self._ratios.items()

  @property
  def num_columns(self):
    """Returns the number of columns of ratio weights.

    A `_RatioWeights` object will usually include multiple "columns" of weights.
    In general, the number of columns is the number of possible "outcomes" of
    the classifier, and the weights in each column are those that will be used
    when the corresponding outcome occurs. In binary classification, for
    example, we'll want one column for positive predictions (used when
    predictions[i] > 0, if we're using the zero-one loss), and another for
    negative predictions (when predictions[i] <= 0). For multiclass problems,
    there will be one column per class.

    Returns:
      The number of columns.
    """
    return self._num_columns

  @classmethod
  def ratio(cls, weights, column_coefficients, numerator_predicate,
            denominator_predicate):
    """Creates a new `_RatioWeights` representing the weights for a ratio.

    This method is used to create the weights for a single ratio, for which
    "numerator_predicate" indicates which examples should be included in the
    numerator of the ratio, and "denominator_predicate" which should be included
    in the denominator.

    Args:
      weights: `DeferredTensor` of example weights.
      column_coefficients: collection of num_columns floats, each of which is
        the quantity by which to multiply the weights in the corresponding
        column.
      numerator_predicate: `Predicate` object representing whether each example
        should be included in the ratio's numerator.
      denominator_predicate: `Predicate` object representing whether each
        example should be included in the ratio's denominator.

    Returns:
      A new `_RatioWeights` representing the ratio.

    Raises:
      TypeError: if any of weights, column_coefficients, numerator_predicate or
        denominator_predicate has the wrong type.
    """
    if not isinstance(weights, deferred_tensor.DeferredTensor):
      raise TypeError("weights must be a DeferredTensor object")
    if isinstance(column_coefficients, deferred_tensor.DeferredTensor):
      raise TypeError("column_coefficients must be a Tensor-like object")
    if not (isinstance(numerator_predicate, predicate.Predicate) and
            isinstance(denominator_predicate, predicate.Predicate)):
      raise TypeError("numerator_predicate and denominator_predictate must "
                      "be Predicate objects")

    column_coefficients = helpers.convert_to_1d_tensor(column_coefficients)
    num_columns = helpers.get_num_elements_of_tensor(column_coefficients)
    key = (weights, denominator_predicate)

    def value_fn(weights_value, predicate_value):
      """Returns the numerator `Tensor`.

      Args:
        weights_value: `Tensor` of example weights.
        predicate_value: indicator `Tensor` representing whether each example
          should be included in the ratio's numerator.

      Returns:
        A `Tensor` containing the element-wise product of the two arguments.

      Raises:
        TypeError: if "weights" is not floating-point.
        ValueError: if "weights" cannot be converted to a rank-1 `Tensor`.
      """
      column = helpers.convert_to_1d_tensor(weights_value, name="weights")
      dtype = column.dtype.base_dtype
      if not dtype.is_floating:
        raise TypeError("weights must be floating-point")
      column *= tf.cast(predicate_value, dtype=dtype)
      row = tf.cast(column_coefficients, dtype=dtype)
      # Return the outer product of the column vector "column" and row vector
      # "row", so that result_ij = column_i * row_j.
      return tf.tensordot(column, row, axes=0)

    # Formerly, we forced the set of examples included in the numerator to be a
    # subset of those in the denominator by using (numerator_predicate.tensor &
    # denominator_predicate.tensor) here. For some rates, however, it's
    # convenient for the numerator to not be a subset of the denominator, so
    # this was removed.
    return _RatioWeights(
        {
            key:
                deferred_tensor.DeferredTensor.apply(value_fn, weights,
                                                     numerator_predicate.tensor)
        }, num_columns)

  def __mul__(self, scalar):
    """Returns the result of multiplying the ratio weights by a scalar."""
    if not isinstance(scalar, numbers.Number):
      raise TypeError("_RatioWeights objects only support *scalar* "
                      "multiplication")

    if scalar == 0:
      return _RatioWeights({}, self._num_columns)
    return _RatioWeights(
        {
            denominator_key: numerator * scalar
            for denominator_key, numerator in six.iteritems(self._ratios)
        }, self._num_columns)

  def __rmul__(self, scalar):
    """Returns the result of multiplying the ratio weights by a scalar."""
    return self.__mul__(scalar)

  def __truediv__(self, scalar):
    """Returns the result of dividing the ratio weights by a scalar."""
    if not isinstance(scalar, numbers.Number):
      raise TypeError("_RatioWeights objects only support *scalar* division")

    if scalar == 0:
      raise ValueError("cannot divide by zero")
    return _RatioWeights(
        {
            denominator_key: numerator / scalar
            for denominator_key, numerator in six.iteritems(self._ratios)
        }, self._num_columns)

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (_RatioWeights / scalar) is allowed, but (scalar / _RatioWeights) is not.

  def __neg__(self):
    """Returns the result of negating the ratio weights."""
    return _RatioWeights(
        {
            denominator_key: -numerator
            for denominator_key, numerator in six.iteritems(self._ratios)
        }, self._num_columns)

  def __add__(self, other):
    """Returns the result of adding two sets of ratio weights."""
    if not isinstance(other, _RatioWeights):
      raise TypeError("_RatioWeights objects can only be added to other "
                      "_RatioWeights objects")

    num_columns = self._num_columns
    if num_columns != other.num_columns:
      raise TypeError("_RatioWeights objects can only be added to other "
                      "_RatioWeights objects with the same number of columns")

    ratios = copy.copy(self._ratios)
    for denominator_key, numerator in other.ratios:
      if denominator_key in ratios:
        ratios[denominator_key] += numerator
      else:
        ratios[denominator_key] = numerator

    return _RatioWeights(ratios, num_columns)

  def __sub__(self, other):
    """Returns the result of subtracting two sets of ratio weights."""
    if not isinstance(other, _RatioWeights):
      raise TypeError("_RatioWeights objects can only be subtracted from other "
                      "_RatioWeights objects")

    num_columns = self._num_columns
    if num_columns != other.num_columns:
      raise TypeError("_RatioWeights objects can only be subtracted from other "
                      "_RatioWeights objects with the same number of columns")

    ratios = copy.copy(self._ratios)
    for denominator_key, numerator in other.ratios:
      if denominator_key in ratios:
        ratios[denominator_key] -= numerator
      else:
        ratios[denominator_key] = -numerator

    return _RatioWeights(ratios, num_columns)

  def _evaluate_denominator(self, denominator, structure_memoizer):
    """Evaluates the denominator portion of a ratio.

    Recall that a `_RatioWeights` object is responsible for computing:
      ratio_weights[j] = weights[j] 1{j in numerator_subset}
          / (mean_i weights[i] 1{i in denominator_subset})
    This method returns (an approximation of) the denominator portion of this
    ratio. The numerator is calculated in the `_RatioWeights`.evaluate method.

    The implementation is complicated by the fact that, although the
    denominators of our ratios should evaluate to "the average weight of the
    examples included in the ratio's denominator", we don't have access to the
    entire dataset (instead, we will typically just get a sequence of
    minibatches). Hence, we can't compute the average weight across the entire
    dataset directly. Instead, we keep running sums of the total weight of
    examples included in the denominator, and the number of examples seen, and
    update them before each minibatch (in the update_ops associated with the
    running sum variables).

    Args:
      denominator: (`DeferredTensor`, `Predicate`) pair, the first being the
        example weights, and the second the predicate indicating which examples
        are included in the denominator.
      structure_memoizer: dict, which memoizes portions of the calculation to
        simplify the resulting TensorFlow graph. It must contain the keys
        "denominator_lower_bound" and "global_step", with the corresponding
        values being the minimum allowed value of a rate denominator (a float),
        and the current iterate (a non-negative integer, starting at zero),
        respectively.

    Returns:
      A `DeferredTensor` containing the (approximate) denominator.
    """
    key = (_RatioWeights, denominator)
    if key not in structure_memoizer:
      # We use double precision arithmetic for the running sums because we
      # don't want numerical errors to ruin our estimates if we perform a very
      # large number of iterations.
      running_dtype = tf.float64

      def update_ops_fn(running_averages_variable, structure_memoizer,
                        value_memoizer):
        """Updates the running sums before each call to the train_op."""
        weights, denominator_predicate = denominator
        weights = helpers.convert_to_1d_tensor(
            weights(structure_memoizer, value_memoizer), name="weights")
        dtype = weights.dtype.base_dtype
        if not dtype.is_floating:
          raise TypeError("weights must be floating-point")

        update_ops = []
        update_ops.append(
            tf.debugging.assert_non_negative(
                weights, message="weights must be non-negative"))

        denominator_weights = weights * tf.cast(
            denominator_predicate.tensor(structure_memoizer, value_memoizer),
            dtype=dtype)

        # We take convex combinations (with parameter running_proportion) to
        # make sure that both running_average_sum and running_average_count
        # are divided by the number of minibatches, as explained below.
        running_proportion = 1.0 / (
            tf.maximum(
                tf.cast(
                    structure_memoizer[defaults.GLOBAL_STEP_KEY],
                    dtype=running_dtype), 0.0) + 1.0)
        running_average_sum = (
            running_averages_variable[0] * (1.0 - running_proportion) +
            tf.cast(tf.reduce_sum(denominator_weights), dtype=running_dtype) *
            running_proportion)
        running_average_count = (
            running_averages_variable[1] * (1.0 - running_proportion) +
            tf.cast(tf.size(denominator_weights), dtype=running_dtype) *
            running_proportion)

        update_ops.append(
            running_averages_variable.assign(
                [running_average_sum, running_average_count]))

        return update_ops

      # The first element of the running_averages variable will contain the sum
      # of the weights included in the denominator that we've seen so far,
      # divided by the number of minibatches that we've seen so far. Similarly,
      # the second element will contain the average size of the minibatches
      # we've seen so far. Their ratio will therefore be the sum of the weights
      # included in the denominator, divided by the number of examples we've
      # seen so far. The reason for dividing both quantities by the number of
      # minibatches is to prevent them from growing without bound during
      # training.
      running_averages = deferred_tensor.DeferredVariable(
          [1.0, 1.0],
          trainable=False,
          name="tfco_running_average_sum_and_count",
          dtype=running_dtype,
          update_ops_fn=update_ops_fn)

      def average_denominator_weight_fn(running_averages_variable):
        """Returns the average denominator weight `Tensor`."""
        # This code calculates max(denominator_lower_bound, running_average_sum
        # / running_average_count) safely, even when running_average_count is
        # zero (including when running_average_sum is also zero, in which case
        # the result will be denominator_lower_bound). We use a tf.cond to make
        # sure that we only perform the division if we know that it will result
        # in a quantity larger than denominator_lower_bound.
        running_denominator_lower_bound = tf.cast(
            structure_memoizer[defaults.DENOMINATOR_LOWER_BOUND_KEY],
            dtype=running_dtype)
        running_average_sum = running_averages_variable[0]
        running_average_count = running_averages_variable[1]
        return tf.cond(
            running_average_count * running_denominator_lower_bound <
            running_average_sum,
            true_fn=lambda: running_average_sum / running_average_count,
            false_fn=lambda: running_denominator_lower_bound)

      structure_memoizer[key] = deferred_tensor.DeferredTensor.apply(
          average_denominator_weight_fn, running_averages)

    return structure_memoizer[key]

  def evaluate(self, structure_memoizer):
    """Computes and returns the `Tensor` of ratio weights.

    Args:
      structure_memoizer: dict, which memoizes portions of the calculation to
        simplify the resulting TensorFlow graph. It must contain the keys
        "denominator_lower_bound" and "global_step", with the corresponding
        values being the minimum allowed value of a rate denominator (a float),
        and the current iterate (a non-negative integer, starting at zero),
        respectively.

    Returns:
      A `DeferredTensor` containing the weights associated with each example.
    """
    ratios = []
    for denominator_key, numerator in six.iteritems(self._ratios):
      denominator = self._evaluate_denominator(denominator_key,
                                               structure_memoizer)

      def ratio_fn(numerator_value, denominator_value):
        """Returns the value of the current ratio as a `Tensor`."""
        dtype = numerator_value.dtype.base_dtype
        return numerator_value / tf.cast(denominator_value, dtype=dtype)

      ratios.append(
          deferred_tensor.DeferredTensor.apply(ratio_fn, numerator,
                                               denominator))

    # It's probably paranoid to call stop_gradient on the ratio weights, but
    # it shouldn't do any harm, and might prevent failure if someone's doing
    # something weird.
    value = deferred_tensor.DeferredTensor.apply(
        lambda *args: tf.stop_gradient(  # pylint: disable=g-long-lambda
            sum(args, np.zeros((1, self._num_columns)))),
        *ratios)

    return value


@six.add_metaclass(abc.ABCMeta)
class Term(helpers.RateObject):
  """Represents a rate term within a rate optimization problem.

  `Term`s support several arithmetic operations: addition, subtraction,
  negation, scalar multiplication and scalar division. Of these, addition and
  subtraction can only be performed on *compatible* terms, i.e. those with
  matching keys.

  This functionality is needed since a `BasicExpression` is a linear combination
  of `Term`s, and implements arithmetic operators for manipulating and combining
  such linear combinations. To ensure that pairwise operations are performed on
  compatible `Term`s, and *only* compatible `Term`s, the `BasicExpression` class
  stores its terms in a dictionary with its keys being the results of the
  `Term`.key method.
  """

  @abc.abstractproperty
  def is_differentiable(self):
    """Returns true only if the loss is {sub,super}differentiable.

    This property is used to check that non-differentiable losses (e.g. the
    zero-one loss) aren't used in contexts in which we will try to optimize over
    them. Non-differentiable losses, however, can be *evaluated* safely.

    Returns:
      True if this `Term`'s loss is {sub,super}differentiable. False otherwise.
    """

  @abc.abstractproperty
  def key(self):
    """A key used to determine whether different `Term`s are compatible.

    The result of this function is used to determine whether two `Term`s are
    compatible: `Term`s with the same key can be added and subtracted, and those
    with different keys cannot. Each key can therefore be identified with a set
    of `Term`s that can be added and subtracted to/from each other.

    Returns:
      An object implementing __eq__. When two `Term`s have the same key, they
      can be safely combined using binary arithmetic operations. When they have
      different keys, the `BasicExpression` owning them will not attempt to do
      so.
    """

  @abc.abstractmethod
  def __mul__(self, scalar):
    """Returns the result of multiplying this `Term` by a scalar."""

  def __rmul__(self, scalar):
    """Returns the result of multiplying this `Term` by a scalar."""
    return self.__mul__(scalar)

  @abc.abstractmethod
  def __truediv__(self, scalar):
    """Returns the result of dividing this `Term` by a scalar."""

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (Term / scalar) is allowed, but (scalar / Term) is not.

  @abc.abstractmethod
  def __neg__(self):
    """Returns the result of negating this `Term`."""

  @abc.abstractmethod
  def __add__(self, other):
    """Returns the result of adding two `Term`s.

    Args:
      other: `Term` with the same key as self.

    Returns:
      A new `Term` representing the sum of self and other.
    """

  @abc.abstractmethod
  def __sub__(self, other):
    """Returns the result of subtracting two `Term`s.

    Args:
      other: `Term` with the same key as self.

    Returns:
      A new `Term` representing the difference of self and other.
    """

  @abc.abstractmethod
  def evaluate(self, structure_memoizer):
    """Computes and returns the value of this `Term`.

    Args:
      structure_memoizer: dict, which memoizes portions of the calculation to
        simplify the resulting TensorFlow graph. It must contain the keys
        "denominator_lower_bound" and "global_step", with the corresponding
        values being the minimum allowed value of a rate denominator (a float),
        and the current iterate (a non-negative integer, starting at zero),
        respectively.

    Returns:
      A `DeferredTensor` containing the value of this `Term`.
    """


class TensorTerm(Term):
  """`Term` object wrapping a `DeferredTensor` or `Tensor`-like object."""

  def __init__(self, tensor):
    """Creates a new `TensorTerm`.

    Args:
      tensor: scalar `DeferredTensor` or `Tensor`-like object that this
        `TensorTerm` will wrap.

    Raises:
      TypeError: if "tensor" is not a `DeferredTensor` or `Tensor`-like object.
    """
    if isinstance(tensor, deferred_tensor.DeferredTensor):
      self._tensor = tensor
    elif not isinstance(tensor, helpers.RateObject):
      self._tensor = deferred_tensor.ExplicitDeferredTensor(tensor)
    else:
      raise TypeError("tensor argument to TensorTerm's constructor should be a "
                      "DeferredTensor or a Tensor-like object")

  @property
  def tensor(self):
    """Returns the `DeferredTensor` wrapped by this `TensorTerm`."""
    return self._tensor

  @property
  def is_differentiable(self):
    """Returns `True`, since we assume that `Tensor`s can be differentiated."""
    return True

  @property
  def key(self):
    """A key used to determine whether different `Term`s are compatible.

    The result of this function is used to determine whether two `Term`s are
    compatible: `Term`s with the same key can be added and subtracted, and those
    with different keys cannot. Each key can therefore be identified with a set
    of `Term`s that can be added and subtracted to/from each other.

    `TensorTerm`s are always compatible with each other (but not with other
    `Term`s). Hence, the key returned by this method is simply the type
    (`TensorTerm`).

    Returns:
      The type (`TensorTerm`).
    """
    return TensorTerm

  def __mul__(self, scalar):
    """Returns the result of multiplying this `TensorTerm` by a scalar."""
    return TensorTerm(self._tensor * scalar)

  def __truediv__(self, scalar):
    """Returns the result of dividing this `TensorTerm` by a scalar."""
    return TensorTerm(self._tensor / scalar)

  def __neg__(self):
    """Returns the result of negating this `TensorTerm`."""
    return TensorTerm(-self._tensor)

  def __add__(self, other):
    """Returns the result of adding two `TensorTerm`s.

    Args:
      other: `TensorTerm` with the same key as self.

    Returns:
      A new `TensorTerm` representing the sum of self and other.
    """
    if self.key != other.key:
      raise ValueError("Terms can only be added if they have the same key")

    return TensorTerm(self._tensor + other.tensor)

  def __sub__(self, other):
    """Returns the result of subtracting two `TensorTerm`s.

    Args:
      other: `TensorTerm` with the same key as self.

    Returns:
      A new `TensorTerm` representing the difference of self and other.
    """
    if self.key != other.key:
      raise ValueError("Terms can only be subtracted if they have the same key")

    return TensorTerm(self._tensor - other.tensor)

  def evaluate(self, structure_memoizer):
    """Computes and returns the value of this `TensorTerm`.

    Args:
      structure_memoizer: dict, which memoizes portions of the calculation to
        simplify the resulting TensorFlow graph. It must contain the keys
        "denominator_lower_bound" and "global_step", with the corresponding
        values being the minimum allowed value of a rate denominator (a float),
        and the current iterate (a non-negative integer, starting at zero),
        respectively.

    Returns:
      A `DeferredTensor` containing the value of this `TensorTerm`.
    """
    return self._tensor


class BinaryClassificationTerm(Term):
  """`Term` object representing a binary classification rate.

  This class represents a rate (and linear combinations of "compatible" rates,
  which will be discussed later). Every rate is represented in terms of positive
  and negative prediction rates on a subset of the dataset:
    ppr = (mean_i weights[i] 1{i in numerator_subset} 1{predictions[i] > 0})
        / (mean_i weights[i] 1{i in denominator_subset})
    npr = (mean_i weights[i] 1{i in numerator_subset} 1{predictions[i] < 0})
        / (mean_i weights[i] 1{i in denominator_subset})
    rate = positive_coefficient * ppr + negative_coefficient * npr
  The "positive_coefficient * ppr" and "negative_coefficient * npr" quantities,
  above, can be written in the form of `_RatioWeights` objects as:
    positive_term = positive_coefficient * ppr
        = (mean_i positive_ratio_weights[i] 1{predictions[i] > 0})
    negative_term = negative_coefficient * ppr
        = (mean_i negative_ratio_weights[i] 1{predictions[i] < 0})
    rate = positive_term + negative_term
  where:
    ratio_weights[j] = weights[j] 1{j in numerator_subset}
        / (mean_i weights[i] 1{i in denominator_subset})
    positive_ratio_weights = positive_coefficient * ratio_weights
    negative_ratio_weights = negative_coefficient * ratio_weights
  A `BinaryClassificationTerm` contains a two-column `_RatioWeights` object with
  the positive and negative ratio weights. When constructed using the "ratio"
  method, the columns of this object will be exactly the
  "positive_ratio_weights" and "negative_ratio_weights" weights defined above
  (things get more complicated, however, once we take linear combinations of
  `BinaryClassificationTerm`s into account).

  In addition to the predictions and weights discussed above, a
  `BinaryClassificationTerm` also has an associated loss object (of type
  `BinaryClassificationLoss`), which is used to approximate the positive and
  negative prediction indicators. For example, if we're using the hinge loss,
  then instead of defining ppr and npr in terms of "1{predictions[i] > 0}" and
  "1{predictions[i] < 0}", a hinge approximation to these indicators will be
  used instead.

  Several arithmetic operations are overloaded: scalar multiplication, scalar
  division, and negation of `BinaryClassificationTerm`s yields new
  `BinaryClassificationTerm`s on which the specified operation has been
  performed. Binary operations (addition and subtraction) are more complicated:
  they can *only* be applied to "compatible" objects, i.e. the objects to be
  added or subtracted must both be `BinaryClassificationTerm`s, both have the
  same set of predictions, and both use the same loss function. If these
  conditions are satisfied, then the `BinaryClassificationTerm`s can be added or
  subtracted by simply adding or subtracting the contained `_RatioWeights`
  objects.

  This "compatibility" condition is checked using the "key" method. If two
  `Term`s have the same keys, then they may be combined using a binary
  arithmetic operation. This is used by the `BasicExpression` object, which
  represents arbitrary linear combinations of `Term`s, to identify compatible
  `Term`s by storing its `Term`s in a dictionary, with its keys taken from the
  "key" method.

  The benefit of combining compatible terms is that, when using a relaxed loss
  (e.g. a hinge) it enables us to use a tighter approximation to the true
  (zero-one based) rate, since we're only performing the loss approximation
  *once*, after we've simplified as much as possible, instead of performing a
  separate approximation for each component rate.
  """

  def __init__(self, predictions, ratio_weights, loss):
    """Creates a new `BinaryClassificationTerm`.

    As explained in the class docstring, a `BinaryClassificationTerm` represents
    a "rate" approximating:
      positive_term = positive_coefficient * ppr
          = (mean_i positive_ratio_weights[i] 1{predictions[i] > 0})
      negative_term = negative_coefficient * ppr
          = (mean_i negative_ratio_weights[i] 1{predictions[i] < 0})
      rate = positive_term + negative_term
    where "positive_ratio_weights" and "negative_ratio_weights" are the two
    columns of the "ratio_weights" parameter to this constructor, and the "loss"
    parameter defines how to approximate the indicators "1{predictions[i] > 0}"
    and "1{predictions[i] < 0}".

    Args:
      predictions: `DeferredTensor` of shape (n,), where n is the number of
        examples.
      ratio_weights: `_RatioWeights` object with two columns, representing the
        per-example weights associated with positive and negative predictions,
        respectively
      loss: `BinaryClassificationLoss`, the loss function to use.

    Raises:
      TypeError: if predictions or ratio_weights has the wrong type.
    """
    if not isinstance(predictions, deferred_tensor.DeferredTensor):
      raise TypeError("predictions must be a DeferredTensor object")
    if not isinstance(ratio_weights, _RatioWeights):
      raise TypeError("ratio_weights must be a _RatioWeights object")

    self._predictions = predictions
    self._ratio_weights = ratio_weights
    self._loss = loss

  @classmethod
  def ratio(cls, positive_coefficient, negative_coefficient, predictions,
            weights, numerator_predicate, denominator_predicate, loss):
    """Creates a new `BinaryClassificationTerm` representing a rate.

    This method is used to create a single rate using the given loss on the
    given predictions, for which "numerator_predicate" indicates which examples
    should be included in the numerator of the ratio, and
    "denominator_predicate" which should be included in the denominator.

    The result of this method represents the "rate":
      ppr = (mean_i weights[i] 1{i in numerator_subset} ploss(predictions[i]))
          / (mean_i weights[i] 1{i in denominator_subset})
      npr = (mean_i weights[i] 1{i in numerator_subset} nloss(predictions[i]))
          / (mean_i weights[i] 1{i in denominator_subset})
      rate = positive_coefficient * ppr + negative_coefficient * npr
    where "ploss(predictions[i])" and "nloss(predictions[i])" represent
    approximations to the indicators "1{predictions[i] > 0}" and
    "1{predictions[i] < 0}, respectively. The particular approximation used
    depends on the "loss" parameter to this method.

    Args:
      positive_coefficient: float, the amount of weight to place on the positive
        predictions.
      negative_coefficient: float, the amount of weight to place on the negative
        predictions.
      predictions: `DeferredTensor` of shape (n,), where n is the number of
        examples.
      weights: `DeferredTensor` of example weights, with shape (m,), where m is
        broadcastable to n.
      numerator_predicate: `Predicate` object representing whether each example
        should be included in the ratio's numerator.
      denominator_predicate: `Predicate` object representing whether each
        example should be included in the ratio's denominator.
      loss: `BinaryClassificationLoss`, the loss function to use.

    Returns:
      A new `BinaryClassificationTerm` representing the rate.

    Raises:
      TypeError: if any of predictions, weights, numerator_predicate, or
        denominator_predicate has the wrong type.
    """
    if not (isinstance(predictions, deferred_tensor.DeferredTensor) and
            isinstance(weights, deferred_tensor.DeferredTensor)):
      raise TypeError("predictions and weights must be DeferredTensor objects")
    if not (isinstance(numerator_predicate, predicate.Predicate) and
            isinstance(denominator_predicate, predicate.Predicate)):
      raise TypeError("numerator_predicate and denominator_predictate must "
                      "be Predicate objects")

    ratio_weights = _RatioWeights.ratio(
        weights, [positive_coefficient, negative_coefficient],
        numerator_predicate, denominator_predicate)
    assert ratio_weights.num_columns == 2

    return BinaryClassificationTerm(predictions, ratio_weights, loss)

  @property
  def predictions(self):
    """Returns the predictions `DeferredTensor`.

    Every `BinaryClassificationTerm` consists of three main parts: the
    predictions (i.e. the model outputs) on which the term is calculated, the
    per-example weights associated with positive and negative predictions (a
    `_RatioWeights` object), and the loss function to use to approximate the
    rate(s). This method returns the predictions.

    Returns:
      rank-1 `DeferredTensor` of per-example predictions.
    """
    return self._predictions

  @property
  def ratio_weights(self):
    """Returns the ratio weights.

    Every `BinaryClassificationTerm` consists of three main parts: the
    predictions (i.e. the model outputs) on which the term is calculated, the
    per-example weights associated with positive and negative predictions (a
    `_RatioWeights` object), and the loss function to use to approximate the
    rate(s). This method returns the `_RatioWeights` object associated with the
    per-example weights.

    Returns:
      `_RatioWeights` object representing the weights on predictions, containing
      two columns, for the positive and negative predictions, respectively.
    """
    return self._ratio_weights

  @property
  def loss(self):
    """Returns the loss.

    Every `BinaryClassificationTerm` consists of three main parts: the
    predictions (i.e. the model outputs) on which the term is calculated, the
    per-example weights associated with positive and negative predictions (a
    `_RatioWeights` object), and the loss function to use to approximate the
    rate(s). This method returns the loss.

    Returns:
      `BinaryClassificationLoss` object associated with this
      `BinaryClassificationTerm`.
    """
    return self._loss

  @property
  def is_differentiable(self):
    """Returns true only if the loss is {sub,super}differentiable.

    This property is used to check that non-differentiable losses (e.g. the
    zero-one loss) aren't used in contexts in which we will try to optimize over
    them. Non-differentiable losses, however, can be *evaluated* safely.

    Returns:
      True if this `BinaryClassificationTerm`'s loss is
      {sub,super}differentiable. False otherwise.
    """
    return self._loss.is_differentiable

  @property
  def key(self):
    """A key used to determine whether different `Term`s are compatible.

    The result of this function is used to determine whether two `Term`s are
    compatible: `Term`s with the same key can be added and subtracted, and those
    with different keys cannot. Each key can therefore be identified with a set
    of `Term`s that can be added and subtracted to/from each other.

    `BinaryClassificationTerm`s are compatible iff they have the same
    predictions and loss--they may, however, have different sets of weights.
    Hence, the key returned by this method contains the predictions and loss, in
    addition to the type (`BinaryClassificationTerm`).

    Returns:
      A 3-tuple, containing the `BinaryClassificationTerm` type, predictions
      `DeferredTensor`, and the loss `BinaryClassificationLoss`.
    """
    return BinaryClassificationTerm, self._predictions, self._loss

  # The Term base class overloads __rmul__, which will fall-back to this
  # function.
  def __mul__(self, scalar):
    """Returns the result of multiplying by a scalar."""
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Term objects only support *scalar* multiplication")

    return BinaryClassificationTerm(self._predictions,
                                    self._ratio_weights * scalar, self._loss)

  def __truediv__(self, scalar):
    """Returns the result of dividing by a scalar."""
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Term objects only support *scalar* division")

    return BinaryClassificationTerm(self._predictions,
                                    self._ratio_weights / scalar, self._loss)

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (Term / scalar) is allowed, but (scalar / Term) is not.

  def __neg__(self):
    """Returns the result of negating this `BinaryClassificationTerm`."""
    return BinaryClassificationTerm(self._predictions, -self._ratio_weights,
                                    self._loss)

  def __add__(self, other):
    """Returns the result of adding two `BinaryClassificationTerm`s.

    Args:
      other: `BinaryClassificationTerm` with the same key as self.

    Returns:
      A new `BinaryClassificationTerm` representing the sum of self and other.

    Raises:
      ValueError: if self and other are not compatible, i.e. if they have
        different keys.
    """
    if self.key != other.key:
      raise ValueError("Terms can only be added if they have the same key")

    return BinaryClassificationTerm(self._predictions,
                                    self._ratio_weights + other.ratio_weights,
                                    self._loss)

  def __sub__(self, other):
    """Returns the result of subtracting two `BinaryClassificationTerm`s.

    Args:
      other: `BinaryClassificationTerm` with the same key as self.

    Returns:
      A new `BinaryClassificationTerm` representing the difference of self and
      other.

    Raises:
      ValueError: if self and other are not compatible, i.e. if they have
        different keys.
    """
    if self.key != other.key:
      raise ValueError("Terms can only be subtracted if they have the same key")

    return BinaryClassificationTerm(self._predictions,
                                    self._ratio_weights - other.ratio_weights,
                                    self._loss)

  def evaluate(self, structure_memoizer):
    """Computes and returns the value of this `BinaryClassificationTerm`.

    Args:
      structure_memoizer: dict, which memoizes portions of the calculation to
        simplify the resulting TensorFlow graph. It must contain the keys
        "denominator_lower_bound" and "global_step", with the corresponding
        values being the minimum allowed value of a rate denominator (a float),
        and the current iterate (a non-negative integer, starting at zero),
        respectively.

    Returns:
      A `DeferredTensor` containing the value of this
      `BinaryClassificationTerm`.
    """
    # Evalaute the weights on the positive and negative approximate indicators.
    ratio_weights = self._ratio_weights.evaluate(structure_memoizer)

    def average_loss_fn(predictions_value, ratio_weights_value):
      """Returns the average loss."""
      losses = self._loss.evaluate_binary_classification(
          predictions_value, ratio_weights_value)
      return tf.reduce_mean(losses)

    return deferred_tensor.DeferredTensor.apply(average_loss_fn,
                                                self._predictions,
                                                ratio_weights)


class MulticlassTerm(Term):
  """`Term` object representing a multiclass classification rate.

  This class represents a rate (and linear combinations of "compatible" rates,
  which will be discussed later). Every rate is represented in terms of
  prediction rates (i.e. how often the classifier predicts class j) on a subset
  of the dataset:
    pr[j] = (mean_i weights[i] 1{i in numerator_subset} *
             1{predictions[i, j] > predictions[i, k] forall k != j})
         / (mean_i weights[i] 1{i in denominator_subset})
  Here, pr[j] is the prediction rate of class j, i.e. the (weighted) proportion
  of "numerator" elements on which we predict class j, divided by the (weighted)
  proportion of denominator elements.  The overall rate is a linear combination
  of these prediction rates (the coefficients are a constructor parameter):
    rate = sum_j coefficient[j] * pr[j]
  We actually represent the above expression a bit differently, internally,
  using a `_RatioWeights` object:
    term[j] = coefficient[j] * pr[j]
        = (mean_i ratio_weights[i, j] *
           1{predictions[i, j] > predictions[i, k] forall k != j})
    rate = sum_j term[j]
  where:
    ratio_weights[i, j] = coefficient[j] * weights[i] * 1{i in numerator_subset}
        / (mean_k weights[k] 1{k in denominator_subset})
  A `MulticlassTerm` contains a `_RatioWeights` object with one column for every
  class (i.e. every possible j, above). When constructed using the "ratio"
  method, the columns of this object will be exactly the "ratio_weights" weights
  defined above (things get more complicated, however, once we take linear
  combinations of `MulticlassTerm`s into account).

  In addition to the predictions and weights discussed above, a `MulticlassTerm`
  also has an associated loss object (of type `MulticlassLoss`), which is used
  to approximate the positive and negative prediction indicators. For example,
  if we're using the hinge loss, then instead of defining ppr and npr in terms
  of "1{predictions[i, j] > predictions[i, k] forall k != j}", a hinge
  approximation to these indicators will be used instead.

  Several arithmetic operations are overloaded: scalar multiplication, scalar
  division, and negation of `MulticlassTerm`s yields new `MulticlassTerm`s on
  which the specified operation has been performed. Binary operations (addition
  and subtraction) are more complicated: they can *only* be applied to
  "compatible" objects, i.e. the objects to be added or subtracted must both be
  `MulticlassTerm`s, both have the same set of predictions, and both use the
  same loss function. If these conditions are satisfied, then the
  `MulticlassTerm`s can be added or subtracted by simply adding or subtracting
  the contained `_RatioWeights` objects.

  This "compatibility" condition is checked using the "key" method. If two
  `Term`s have the same keys, then they may be combined using a binary
  arithmetic operation. This is used by the `BasicExpression` object, which
  represents arbitrary linear combinations of `Term`s, to identify compatible
  `Term`s by storing its `Term`s in a dictionary, with its keys taken from the
  "key" method.

  The benefit of combining compatible terms is that, when using a relaxed loss
  (e.g. a hinge) it enables us to use a tighter approximation to the true
  (zero-one based) rate, since we're only performing the loss approximation
  *once*, after we've simplified as much as possible, instead of performing a
  separate approximation for each component rate.
  """

  def __init__(self, predictions, ratio_weights, loss):
    """Creates a new `MulticlassTerm`.

    As explained in the class docstring, a `MulticlassTerm` represents a "rate"
    approximating:
      term[j] = coefficient[j] * pr[j]
          = (mean_i ratio_weights[i, j] *
             1{predictions[i, j] > predictions[i, k] forall k != j})
      rate = sum_j term[j]
    where "ratio_weights" is a parameter to this constructor, with one column j
    per class, and the "loss" parameter defines how to approximate the
    indicators "1{predictions[i] > 0}" and "1{predictions[i] < 0}".

    Args:
      predictions: `DeferredTensor` of shape (n,), where n is the number of
        examples.
      ratio_weights: `_RatioWeights` object with one column per class,
        representing the per-example weights associated with predictions of this
        class.
      loss: `MulticlassLoss`, the loss function to use.

    Raises:
      TypeError: if predictions or ratio_weights has the wrong type.
    """
    if not isinstance(predictions, deferred_tensor.DeferredTensor):
      raise TypeError("predictions must be a DeferredTensor object")
    if not isinstance(ratio_weights, _RatioWeights):
      raise TypeError("ratio_weights must be a _RatioWeights object")

    self._predictions = predictions
    self._ratio_weights = ratio_weights
    self._loss = loss

  @classmethod
  def ratio(cls, coefficients, predictions, weights, numerator_predicate,
            denominator_predicate, loss):
    """Creates a new `MulticlassTerm` representing a rate.

    This method is used to create a single rate using the given loss on the
    given predictions, for which "numerator_predicate" indicates which examples
    should be included in the numerator of the ratio, and
    "denominator_predicate" which should be included in the denominator.

    The result of this method represents the "rate":
      ppr = (mean_i weights[i] 1{i in numerator_subset} ploss(predictions[i]))
          / (mean_i weights[i] 1{i in denominator_subset})
      npr = (mean_i weights[i] 1{i in numerator_subset} nloss(predictions[i]))
          / (mean_i weights[i] 1{i in denominator_subset})
      rate = positive_coefficient * ppr + negative_coefficient * npr
      pr[j] = (mean_i weights[i] 1{i in numerator_subset} *
               lossj(predictions[i, :]))
           / (mean_i weights[i] 1{i in denominator_subset})
      rate = sum_j coefficient[j] * pr[j]
    where "lossj(predictions[i, :])" represents an approximation to the
    indicator "1{predictions[i, j] > predictions[i, k] forall k != j}".  The
    particular approximation used depends on the "loss" parameter to this
    method.

    Args:
      coefficients: collection of floats, one per class, containing the amount
        of weight to place on the predictions of this class.
      predictions: `DeferredTensor` of shape (n,k), where n is the number of
        examples and k is the number of classes.
      weights: `DeferredTensor` of example weights, with shape (m,), where m is
        broadcastable to n.
      numerator_predicate: `Predicate` object representing whether each example
        should be included in the ratio's numerator.
      denominator_predicate: `Predicate` object representing whether each
        example should be included in the ratio's denominator.
      loss: `MulticlassLoss`, the loss function to use.

    Returns:
      A new `MulticlassTerm` representing the rate.

    Raises:
      TypeError: if any of predictions, weights, numerator_predicate, or
        denominator_predicate has the wrong type.
    """
    if not (isinstance(predictions, deferred_tensor.DeferredTensor) and
            isinstance(weights, deferred_tensor.DeferredTensor)):
      raise TypeError("predictions and weights must be DeferredTensor objects")
    if not (isinstance(numerator_predicate, predicate.Predicate) and
            isinstance(denominator_predicate, predicate.Predicate)):
      raise TypeError("numerator_predicate and denominator_predictate must "
                      "be Predicate objects")

    ratio_weights = _RatioWeights.ratio(weights, coefficients,
                                        numerator_predicate,
                                        denominator_predicate)

    return MulticlassTerm(predictions, ratio_weights, loss)

  @property
  def predictions(self):
    """Returns the predictions `DeferredTensor`.

    Every `MulticlassTerm` consists of three main parts: the predictions (i.e.
    the model outputs) on which the term is calculated, the per-example weights
    associated with predictions (a `_RatioWeights` object), and the loss
    function to use to approximate the rate(s). This method returns the
    predictions.

    Returns:
      `DeferredTensor` of per-example predictions with shape (n,k), where n is
      the number of examples and k is the number of classes.
    """
    return self._predictions

  @property
  def ratio_weights(self):
    """Returns the ratio weights.

    Every `MulticlassTerm` consists of three main parts: the predictions (i.e.
    the model outputs) on which the term is calculated, the per-example weights
    associated with predictions (a `_RatioWeights` object), and the loss
    function to use to approximate the rate(s). This method returns the
    `_RatioWeights` object associated with the per-example weights.

    Returns:
      `_RatioWeights` object representing the weights on predictions, containing
      one column per class.
    """
    return self._ratio_weights

  @property
  def loss(self):
    """Returns the loss.

    Every `MulticlassTerm` consists of three main parts: the predictions (i.e.
    the model outputs) on which the term is calculated, the per-example weights
    associated with predictions (a `_RatioWeights` object), and the loss
    function to use to approximate the rate(s). This method returns the loss.

    Returns:
      `MulticlassLoss` object associated with this `MulticlassTerm`.
    """
    return self._loss

  @property
  def is_differentiable(self):
    """Returns true only if the loss is {sub,super}differentiable.

    This property is used to check that non-differentiable losses (e.g. the
    zero-one loss) aren't used in contexts in which we will try to optimize over
    them. Non-differentiable losses, however, can be *evaluated* safely.

    Returns:
      True if this `MulticlassTerm`'s loss is {sub,super}differentiable. False
      otherwise.
    """
    return self._loss.is_differentiable

  @property
  def key(self):
    """A key used to determine whether different `Term`s are compatible.

    The result of this function is used to determine whether two `Term`s are
    compatible: `Term`s with the same key can be added and subtracted, and those
    with different keys cannot. Each key can therefore be identified with a set
    of `Term`s that can be added and subtracted to/from each other.

    `MulticlassTerm`s are compatible iff they have the same predictions and
    loss--they may, however, have different sets of weights. Hence, the key
    returned by this method contains the predictions and loss, in addition to
    the type (`MulticlassTerm`). We also include the number of classes as an
    extra safety check (although it's redundant with the predictions).

    Returns:
      A 4-tuple, containing the `MulticlassTerm` type, predictions
      `DeferredTensor`, number of classes, and the loss `MulticlassLoss`.
    """
    return (MulticlassTerm, self._predictions, self._ratio_weights.num_columns,
            self._loss)

  # The Term base class overloads __rmul__, which will fall-back to this
  # function.
  def __mul__(self, scalar):
    """Returns the result of multiplying by a scalar."""
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Term objects only support *scalar* multiplication")

    return MulticlassTerm(self._predictions, self._ratio_weights * scalar,
                          self._loss)

  def __truediv__(self, scalar):
    """Returns the result of dividing by a scalar."""
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Term objects only support *scalar* division")

    return MulticlassTerm(self._predictions, self._ratio_weights / scalar,
                          self._loss)

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (Term / scalar) is allowed, but (scalar / Term) is not.

  def __neg__(self):
    """Returns the result of negating this `MulticlassTerm`."""
    return MulticlassTerm(self._predictions, -self._ratio_weights, self._loss)

  def __add__(self, other):
    """Returns the result of adding two `MulticlassTerm`s.

    Args:
      other: `MulticlassTerm` with the same key as self.

    Returns:
      A new `MulticlassTerm` representing the sum of self and other.

    Raises:
      ValueError: if self and other are not compatible, i.e. if they have
        different keys.
    """
    if self.key != other.key:
      raise ValueError("Terms can only be added if they have the same key")

    return MulticlassTerm(self._predictions,
                          self._ratio_weights + other.ratio_weights, self._loss)

  def __sub__(self, other):
    """Returns the result of subtracting two `MulticlassTerm`s.

    Args:
      other: `MulticlassTerm` with the same key as self.

    Returns:
      A new `MulticlassTerm` representing the difference of self and
      other.

    Raises:
      ValueError: if self and other are not compatible, i.e. if they have
        different keys.
    """
    if self.key != other.key:
      raise ValueError("Terms can only be subtracted if they have the same key")

    return MulticlassTerm(self._predictions,
                          self._ratio_weights - other.ratio_weights, self._loss)

  def evaluate(self, structure_memoizer):
    """Computes and returns the value of this `MulticlassTerm`.

    Args:
      structure_memoizer: dict, which memoizes portions of the calculation to
        simplify the resulting TensorFlow graph. It must contain the keys
        "denominator_lower_bound" and "global_step", with the corresponding
        values being the minimum allowed value of a rate denominator (a float),
        and the current iterate (a non-negative integer, starting at zero),
        respectively.

    Returns:
      A `DeferredTensor` containing the value of this `MulticlassTerm`.
    """
    # Evalaute the weights on the positive and negative approximate indicators.
    ratio_weights = self._ratio_weights.evaluate(structure_memoizer)

    def average_loss_fn(predictions_value, ratio_weights_value):
      """Returns the average loss."""
      losses = self._loss.evaluate_multiclass(predictions_value,
                                              ratio_weights_value)
      return tf.reduce_mean(losses)

    return deferred_tensor.DeferredTensor.apply(average_loss_fn,
                                                self._predictions,
                                                ratio_weights)
