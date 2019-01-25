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
object that actually represents linear combinations of `Term`s and `Tensor`s,
and it contains some additional logic for combining "compatible" `Term`s.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy
import six
import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import helpers


class _RatioWeights(object):
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

  class EvaluationContext(object):
    """Evaluation context for `_RatioWeights` class.

    A rate constraints problem is constructed as an objective function and set
    of constraints, all of which are represented as `Expression`s, each of which
    contains two `BasicExpression`s, each of which is a linear combination of
    `Term`s. Often, the same `Term` will be included in multiple
    `BasicExpression`s, which normally would result in the construction of a
    TensorFlow graph with a lot of redundant calculations.

    We would prefer the more expensive parts of the graph to be shared when they
    occur multiple times. To accomplish this, we use an `EvaluationContext`,
    which remembers (some) `Variable`s and `Operation`s that have already been
    created, and re-uses them, instead of recreating them each time they're
    needed.

    In the specific case of the `_RatioWeights.EvaluationContext`, the memoized
    quantities are the denominators of the ratios.
    """

    def __init__(self, denominator_lower_bound, global_step):
      """Constructs a new `EvaluationContext`.

      Args:
        denominator_lower_bound: float, smallest allowed value of the
          denominator of a ratio.
        global_step: `Tensor`, the number of iterations that have been performed
          so far (starting at zero).
      """
      self._denominator_lower_bound = denominator_lower_bound
      self._global_step = global_step
      self._denominators = {}

    def evaluate_denominator(self, denominator):
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
      update them before each minibatch (in the set of `Operation`s returned by
      this method).

      Args:
        denominator: (`Tensor`, `Predicate`) pair, the first being the example
          weights, and the second the predicate indicating which examples are
          included in the denominator.

      Returns:
        A (`Tensor`, set, set) tuple containing the (approximate) denominator, a
        set of `Operation`s that should be executed before each training step
        (to update the internal state upon which the denominator depends), and a
        set of `Operation`s that can be executed to re-initialize this state.

      Raises:
        TypeError: if "weights" is not floating-point.
        ValueError: if "weights" cannot be converted to a rank-1 `Tensor`.
      """
      if denominator not in self._denominators:
        weights, denominator_predicate = denominator
        weights = helpers.convert_to_1d_tensor(weights, name="weights")
        dtype = weights.dtype.base_dtype
        if not dtype.is_floating:
          raise TypeError("weights must be floating-point")

        pre_train_ops = set()
        pre_train_ops.add(
            tf.assert_non_negative(
                weights, message="weights must be non-negative"))

        denominator_weights = weights * tf.cast(
            denominator_predicate.predicate, dtype=dtype)

        # The running_average_sum variable will contain the sum of the weights
        # included in the denominator that we've seen so far, divided by the
        # number of minibatches that we've seen so far. Similarly,
        # running_average_count will contain the average size of the minibatches
        # we've seen so far. Their ratio will therefore be the sum of the
        # weights included in the denominator, divided by the number of examples
        # we've seen so far. The reason for dividing both quantities by the
        # number of minibatches is to prevent them from growing without bound
        # during training.
        #
        # We use double precision arithmetic for the running sums because we
        # don't want numerical errors to ruin our estimates if we perform a very
        # large number of iterations.
        running_dtype = tf.float64
        running_average_sum = tf.Variable(
            1.0,
            trainable=False,
            dtype=running_dtype,
            name="running_average_sum")
        running_average_count = tf.Variable(
            1.0,
            trainable=False,
            dtype=running_dtype,
            name="running_average_count")

        # To restart the denominator calculations, we set the two running
        # averages to their initial values.
        restart_ops = set([
            tf.assign(running_average_sum, 1.0),
            tf.assign(running_average_count, 1.0)
        ])

        # We take convex combinations (with parameter running_proportion) to
        # make sure that both running_average_sum and running_average_count are
        # divided by the number of minibatches, as explained above.
        running_proportion = 1.0 / (
            tf.maximum(tf.cast(self._global_step, dtype=running_dtype), 0.0) +
            1.0)
        pre_train_ops.add(
            tf.assign(
                running_average_sum,
                running_average_sum * (1.0 - running_proportion) + tf.cast(
                    tf.reduce_sum(denominator_weights), dtype=running_dtype) *
                running_proportion))
        pre_train_ops.add(
            tf.assign(
                running_average_count,
                running_average_count * (1.0 - running_proportion) +
                tf.cast(tf.size(denominator_weights),
                        dtype=running_dtype) * running_proportion))

        # This code calculates max(denominator_lower_bound, running_average_sum
        # / running_average_count) safely, even when running_average_count is
        # zero (including when running_average_sum is also zero, in which case
        # the result will be denominator_lower_bound). We use a tf.cond to make
        # sure that we only perform the division if we know that it will result
        # in a quantity larger than denominator_lower_bound.
        running_denominator_lower_bound = tf.cast(
            self._denominator_lower_bound, dtype=running_dtype)
        average_denominator_weight = tf.cond(
            running_average_count * running_denominator_lower_bound <
            running_average_sum,
            true_fn=lambda: running_average_sum / running_average_count,
            false_fn=lambda: running_denominator_lower_bound)

        self._denominators[denominator] = (average_denominator_weight,
                                           pre_train_ops, restart_ops)

      return self._denominators[denominator]

  def __init__(self, dtype, ratios):
    """Creates a new `_RatioWeights` object.

    A `RatioWeights` object is a sum of ratios, represented internally as a dict
    mapping denominators to numerators. Each element of this dict represents a
    ratio, and the weights Tensor represented by this object is the sum of these
    ratios. This representation allows us to add ratios with common denominators
    by simply adding their numerators.

    Args:
      dtype: tf.DType, the dtype of the ratio weights.
      ratios: A dict mapping denominators to numerators. Each denominator is a
        (`Tensor`, `Predicate`) pair, the first element being the example
        weights, and the second the predicate indicating which examples are
        included in the denominator. Each numerator is a `Tensor` of example
        weights.
    """
    self._dtype = dtype
    self._ratios = ratios

  @property
  def dtype(self):
    """Returns the `tf.DType` of the result of evaluate()."""
    return self._dtype

  @property
  def ratios(self):
    """Returns the sum of ratios represented by this object in a list.

    Returns:
      A list of (denominator, numerator) pairs. Each denominator is a (`Tensor`,
      `Predicate`) pair, the first element being the example weights, and the
      second the predicate indicating which examples are included in the
      denominator. Each numerator is a `Tensor` of example weights.
    """
    return self._ratios.items()

  @classmethod
  def ratio(cls, weights, numerator_predicate, denominator_predicate):
    """Creates a new `_RatioWeights` representing the weights for a ratio.

    This method is used to create the weights for a single ratio, for which
    "numerator_predicate" indicates which examples should be included in the
    numerator of the ratio, and "denominator_predicate" which should be included
    in the denominator.

    The numerator and denominator predicates can be arbitrary indicator
    `Tensor`s, but the numerator subset will be intersected with the
    denominator's before the rate is calculated.

    Args:
      weights: `Tensor` of example weights.
      numerator_predicate: boolean indicator `Tensor` representing whether each
        example should be included in the ratio's numerator.
      denominator_predicate: boolean indicator `Tensor` representing whether
        each example should be included in the ratio's denominator.

    Returns:
      A new `_RatioWeights` representing the ratio.

    Raises:
      TypeError: if "weights" is not floating-point.
      ValueError: if "weights" cannot be converted to a rank-1 `Tensor`.
    """
    key = (weights, denominator_predicate)

    value = helpers.convert_to_1d_tensor(weights, name="weights")
    dtype = value.dtype.base_dtype
    if not dtype.is_floating:
      raise TypeError("weights must be floating-point")
    # Notice that we force the set of examples included in the numerator to be a
    # subset of those in the denominator. It'll actually work fine either way,
    # but this should be closer to what users expect.
    value *= tf.cast(
        (numerator_predicate & denominator_predicate).predicate, dtype=dtype)

    return _RatioWeights(dtype, {key: value})

  def __mul__(self, scalar):
    """Returns the result of multiplying the ratio weights by a scalar."""
    # Ideally, we'd check that "scalar" is a scalar Tensor, or is a type that
    # can be converted to a scalar Tensor. Unfortunately, this includes a lot of
    # possible types, so the easiest solution would be to actually perform the
    # conversion, and then check that the resulting Tensor has only one element.
    # This, however, would add a dummy element to the Tensorflow graph, and
    # wouldn't work for a Tensor with an unknown size. Hence, we check only the
    # most common failure case (multiplication of two _RatioWeights objects).
    if isinstance(scalar, _RatioWeights):
      raise TypeError(
          "_RatioWeights objects only support *scalar* multiplication: you "
          "cannot multiply two _RatioWeights objects")

    if scalar == 0:
      return _RatioWeights(self._dtype, {})
    return _RatioWeights(
        self._dtype, {
            denominator: numerator * scalar
            for denominator, numerator in six.iteritems(self._ratios)
        })

  def __rmul__(self, scalar):
    """Returns the result of multiplying the ratio weights by a scalar."""
    return self.__mul__(scalar)

  def __truediv__(self, scalar):
    """Returns the result of dividing the ratio weights by a scalar."""
    # We check that "scalar" is not a _RatioWeights object, instead of checking
    # that it is a scalar, for the same reason as in __mul__.
    if isinstance(scalar, _RatioWeights):
      raise TypeError("_RatioWeights objects only support *scalar* division: "
                      "you cannot divide two _RatioWeights objects")

    if scalar == 0:
      raise ValueError("cannot divide by zero")
    return _RatioWeights(
        self._dtype, {
            denominator: numerator / scalar
            for denominator, numerator in six.iteritems(self._ratios)
        })

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (_RatioWeights / scalar) is allowed, but (scalar / _RatioWeights) is not.

  def __neg__(self):
    """Returns the result of negating the ratio weights."""
    return _RatioWeights(
        self._dtype, {
            denominator: -numerator
            for denominator, numerator in six.iteritems(self._ratios)
        })

  def __add__(self, other):
    """Returns the result of adding two sets of ratio weights."""
    if not isinstance(other, _RatioWeights):
      raise TypeError("_RatioWeights objects can only be added to other "
                      "_RatioWeights objects")
    if self._dtype != other.dtype:
      raise TypeError("can only add weights with the same dtypes")

    ratios = copy.copy(self._ratios)
    for denominator, numerator in other.ratios:
      if denominator in ratios:
        ratios[denominator] += numerator
      else:
        ratios[denominator] = numerator

    return _RatioWeights(self._dtype, ratios)

  def __sub__(self, other):
    """Returns the result of subtracting two sets of ratio weights."""
    if not isinstance(other, _RatioWeights):
      raise TypeError("_RatioWeights objects can only be subtracted from "
                      "other _RatioWeights objects")
    if self._dtype != other.dtype:
      raise TypeError("can only subtract weights with the same dtypes")

    ratios = copy.copy(self._ratios)
    for denominator, numerator in other.ratios:
      if denominator in ratios:
        ratios[denominator] -= numerator
      else:
        ratios[denominator] = -numerator

    return _RatioWeights(self._dtype, ratios)

  def evaluate(self, evaluation_context):
    """Computes and returns the `Tensor` of ratio weights.

    Args:
      evaluation_context: `_RatioWeights.EvaluationContext`, which memoizes
        portions of the calculation to simplify the resulting TensorFlow graph.

    Returns:
      A (`Tensor`, set) tuple containing the weights associated with each
      example, a set of `Operation`s that should be executed before each
      training step (to update the internal state upon which the weight
      denominators depend), and a set of `Operation`s that can be executed to
      re-initialize this state.
    """
    value = tf.zeros(1, dtype=self._dtype)
    pre_train_ops = set()
    restart_ops = set()

    for denominator, numerator in six.iteritems(self._ratios):
      denominator_value, denominator_pre_train_ops, denominator_restart_ops = (
          evaluation_context.evaluate_denominator(denominator))
      # The numerator should already be the correct dtype--this cast just makes
      # extra sure.
      value += tf.cast(
          numerator, dtype=self._dtype) / tf.cast(
              denominator_value, dtype=self._dtype)
      pre_train_ops.update(denominator_pre_train_ops)
      restart_ops.update(denominator_restart_ops)

    # It's probably paranoid to call stop_gradient on the ratio weights, but it
    # shouldn't do any harm, and might prevent failure if someone's doing
    # something weird.
    return tf.stop_gradient(value), pre_train_ops, restart_ops


@six.add_metaclass(abc.ABCMeta)
class Term(object):
  """Represents a rate term within a rate optimization problem.

  `Term`s support several arithmetic operations: addition, subtraction,
  negation, scalar multiplication and scalar division. Of these, addition and
  subtraction can only be performed on *compatible* terms, i.e. those with
  matching keys.

  This functionality is needed since a `BasicExpression` is a linear combination
  of `Term`s and `Tensor`s, and implements arithmetic operators for manipulating
  and combining such linear combinations. To ensure that pairwise operations are
  performed on compatible `Term`s, and *only* compatible `Term`s, the
  `BasicExpression` class stores its terms in a dictionary with its keys being
  the results of the `Term`.key method.
  """

  # In the future, we might want Term to have its own EvaluationContexts, so
  # that Terms can perform their own memoization. At the moment, however, only
  # _RatioWeights creates variables and operations, and therefore needs an
  # EvaluationContext, so Term just grabs its EvaluationContext from
  # _RatioWeights.
  class EvaluationContext(_RatioWeights.EvaluationContext):
    """Evaluation context for `Term` class.

    A rate constraints problem is constructed as an objective function and set
    of constraints, all of which are represented as `Expression`s, each of which
    contains two `BasicExpression`s, each of which is a linear combination of
    `Term`s. Often, the same `Term` will be included in multiple
    `BasicExpression`s, which normally would result in the construction of a
    TensorFlow graph with a lot of redundant calculations.

    We would prefer the more expensive parts of the graph to be shared when they
    occur multiple times. To accomplish this, we use an `EvaluationContext`,
    which remembers (some) `Variable`s and `Operation`s that have already been
    created, and re-uses them, instead of recreating them each time they're
    needed.
    """

  @abc.abstractproperty
  def dtype(self):
    """Returns the `tf.DType` of the result of evaluate()."""

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
      An object implementing __hash__ and __eq__. When two `Term`s have the same
      key, they can be safely combined using binary arithmetic operations. When
      they have different keys, the `BasicExpression` owning them will not
      attempt to do so.
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
  def evaluate(self, evaluation_context):
    """Computes and returns the value of this `Term`.

    Args:
      evaluation_context: `Term.EvaluationContext`, which memoizes portions of
        the calculation to simplify the resulting TensorFlow graph.

    Returns:
      A (`Tensor`, set, set) tuple containing the value of this `Term`, a set of
      `Operation`s that should be executed before each training step (to update
      the internal state upon which the `Term` evaluation depends), and a set of
      `Operation`s that can be executed to re-initialize this state.
    """


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
  A `BinaryClassificationTerm` contains the two `_RatioWeights` objects. When
  constructed using the "ratio" method, these two objects will be exactly the
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
  represents arbitrary linear combinations of `Term`s and `Tensor`s, to identify
  compatible `Term`s by storing its `Term`s in a dictionary, with its keys taken
  from the "key" method.

  The benefit of combining compatible terms is that, when using a relaxed loss
  (e.g. a hinge) it enables us to use a tighter approximation to the true
  (zero-one based) rate, since we're only performing the loss approximation
  *once*, after we've simplified as much as possible, instead of performing a
  separate approximation for each component rate.
  """

  def __init__(self, predictions, positive_ratio_weights,
               negative_ratio_weights, loss):
    """Creates a new `BinaryClassificationTerm`.

    As explained in the class docstring, a `BinaryClassificationTerm` represents
    a "rate" approximating as:
      positive_term = positive_coefficient * ppr
          = (mean_i positive_ratio_weights[i] 1{predictions[i] > 0})
      negative_term = negative_coefficient * ppr
          = (mean_i negative_ratio_weights[i] 1{predictions[i] < 0})
      rate = positive_term + negative_term
    where "positive_ratio_weights" and "negative_ratio_weights" are parameters
    to this constructor, and the "loss" parameter defines how to approximate
    the indicators "1{predictions[i] > 0}" and "1{predictions[i] < 0}".

    Args:
      predictions: `Tensor` of shape (n,), where n is the number of examples.
      positive_ratio_weights: `_RatioWeights` object representing the
        per-example weights associated with positive predictions.
      negative_ratio_weights: `_RatioWeights` object representing the
        per-example weights associated with negative predictions.
      loss: `BinaryClassificionLoss`, the loss function to use.
    """
    self._dtype = predictions.dtype.base_dtype
    self._predictions = predictions
    self._positive_ratio_weights = positive_ratio_weights
    self._negative_ratio_weights = negative_ratio_weights
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

    The numerator and denominator predicates can be arbitrary indicator
    `Tensor`s, but the numerator subset will be intersected with the
    denominator's before the rate is calculated.

    Args:
      positive_coefficient: float, the amount of weight to place on the positive
        predictions.
      negative_coefficient: float, the amount of weight to place on the negative
        predictions.
      predictions: `Tensor` of shape (n,), where n is the number of examples.
      weights: `Tensor` of example weights, with shape (m,), where m is
        broadcastable to n.
      numerator_predicate: boolean indicator `Tensor` representing whether each
        example should be included in the ratio's numerator.
      denominator_predicate: boolean indicator `Tensor` representing whether
        each example should be included in the ratio's denominator.
      loss: `BinaryClassificionLoss`, the loss function to use.

    Returns:
      A new `BinaryClassificationTerm` representing the rate.
    """
    ratio_weights = _RatioWeights.ratio(weights, numerator_predicate,
                                        denominator_predicate)
    return BinaryClassificationTerm(predictions,
                                    ratio_weights * positive_coefficient,
                                    ratio_weights * negative_coefficient, loss)

  @property
  def dtype(self):
    """Returns the `tf.DType` of the result of evaluate()."""
    return self._dtype

  @property
  def predictions(self):
    """Returns the predictions `Tensor`.

    Every `BinaryClassificationTerm` consists of three main parts: the
    predictions (i.e. the model outputs) on which the term is calculated, the
    per-example weights associated with positive and negative predictions (both
    are `_RatioWeights` objects), and the loss function to use to approximate
    the rate(s). This method returns the predictions.

    Returns:
      rank-1 `Tensor` of per-example predictions.
    """
    return self._predictions

  @property
  def positive_ratio_weights(self):
    """Returns the positive ratio weights.

    Every `BinaryClassificationTerm` consists of three main parts: the
    predictions (i.e. the model outputs) on which the term is calculated, the
    per-example weights associated with positive and negative predictions (both
    are `_RatioWeights` objects), and the loss function to use to approximate
    the rate(s). This method returns the `_RatioWeights` object associated with
    positive predictions.

    Returns:
      `_RatioWeights` object representing the weights on positive predictions.
    """
    return self._positive_ratio_weights

  @property
  def negative_ratio_weights(self):
    """Returns the negative ratio weights.

    Every `BinaryClassificationTerm` consists of three main parts: the
    predictions (i.e. the model outputs) on which the term is calculated, the
    per-example weights associated with positive and negative predictions (both
    are `_RatioWeights` objects), and the loss function to use to approximate
    the rate(s). This method returns the `_RatioWeights` object associated with
    negative predictions.

    Returns:
      `_RatioWeights` object representing the weights on negative predictions.
    """
    return self._negative_ratio_weights

  @property
  def loss(self):
    """Returns the loss.

    Every `BinaryClassificationTerm` consists of three main parts: the
    predictions (i.e. the model outputs) on which the term is calculated, the
    per-example weights associated with positive and negative predictions (both
    are `_RatioWeights` objects), and the loss function to use to approximate
    the rate(s). This method returns the loss.

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
    predictions and loss---they may, however, have different sets of weights.
    Hence, the key returned by this method contains the predictions and loss, in
    addition to the type (`BinaryClassificationTerm`).

    Returns:
      A 3-tuple, containing the `BinaryClassificationTerm` type, predictions
      `Tensor`, and the loss `BinaryClassificationLoss`.
    """
    return (BinaryClassificationTerm, self._predictions, self._loss)

  # The Term base class overloads __rmul__, which will fall-back to this
  # function.
  def __mul__(self, scalar):
    """Returns the result of multiplying by a scalar."""
    # Ideally, we'd check that "scalar" is a scalar Tensor, or is a type that
    # can be converted to a scalar Tensor. Unfortunately, this includes a lot of
    # possible types, so the easiest solution would be to actually perform the
    # conversion, and then check that the resulting Tensor has only one element.
    # This, however, would add a dummy element to the Tensorflow graph, and
    # wouldn't work for a Tensor with an unknown size. Hence, we check only the
    # most common failure case (multiplication of two Terms).
    if isinstance(scalar, Term):
      raise TypeError("Term objects only support *scalar* multiplication: "
                      "you cannot multiply two Terms")

    return BinaryClassificationTerm(
        self._predictions, self._positive_ratio_weights * scalar,
        self._negative_ratio_weights * scalar, self._loss)

  def __truediv__(self, scalar):
    """Returns the result of dividing by a scalar."""
    # We check that "scalar" is not a Term, instead of checking that it is a
    # scalar, for the same reason as in __mul__.
    if isinstance(scalar, Term):
      raise TypeError("Term objects only support *scalar* division: you "
                      "cannot divide two Terms")

    return BinaryClassificationTerm(
        self._predictions, self._positive_ratio_weights / scalar,
        self._negative_ratio_weights / scalar, self._loss)

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (Term / scalar) is allowed, but (scalar / Term) is not.

  def __neg__(self):
    """Returns the result of negating this `BinaryClassificationTerm`."""
    return BinaryClassificationTerm(self._predictions,
                                    -self._positive_ratio_weights,
                                    -self._negative_ratio_weights, self._loss)

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
      raise ValueError("binary terms can only be added if they have the same "
                       "key")

    return BinaryClassificationTerm(
        self._predictions,
        self._positive_ratio_weights + other.positive_ratio_weights,
        self._negative_ratio_weights + other.negative_ratio_weights, self._loss)

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
      raise ValueError("binary terms can only be subtracted if they have the "
                       "same key")

    return BinaryClassificationTerm(
        self._predictions,
        self._positive_ratio_weights - other.positive_ratio_weights,
        self._negative_ratio_weights - other.negative_ratio_weights, self._loss)

  def evaluate(self, evaluation_context):
    """Computes and returns the value of this `BinaryClassificationTerm`.

    Args:
      evaluation_context: `Term.EvaluationContext`, which memoizes portions of
        the calculation to simplify the resulting TensorFlow graph.

    Returns:
      A (`Tensor`, set, set) tuple containing the value of this
      `BinaryClassificationTerm`, a set of `Operation`s that should be executed
      before each training step (to update the internal state upon which the
      `BinaryClassificationTerm` evaluation depends), and a set of `Operation`s
      that can be executed to re-initialize this state.
    """
    pre_train_ops = set()
    restart_ops = set()

    # Evalaute the weights on the positive and negative approximate indicators.
    positive_weights, positive_pre_train_ops, positive_restart_ops = (
        self._positive_ratio_weights.evaluate(evaluation_context))
    negative_weights, negative_pre_train_ops, negative_restart_ops = (
        self._negative_ratio_weights.evaluate(evaluation_context))
    pre_train_ops.update(positive_pre_train_ops)
    pre_train_ops.update(negative_pre_train_ops)
    restart_ops.update(positive_restart_ops)
    restart_ops.update(negative_restart_ops)

    # Use broadcasting to make the positive_weights and negative_weights Tensors
    # have the same shape (yes, this is sort of hacky). The _RatioWeights object
    # has already checked that they're both rank-1, so this code just makes sure
    # that they're the same size before attempting to stack them.
    positive_weights += tf.zeros_like(negative_weights)
    negative_weights += tf.zeros_like(positive_weights)

    weights = tf.stack([positive_weights, negative_weights], axis=1)
    losses = self._loss.evaluate_binary_classification(self._predictions,
                                                       weights)
    # If losses isn't one-dimensional, then something has gone badly wrong---we
    # should have checked all of the dimensions before reaching this point.
    # Likewise, loss functions are required to return a Tensor of the same dtype
    # as the predictions.
    pre_train_ops.add(
        tf.assert_rank(losses, 1, message="losses must be one-dimensional"))
    pre_train_ops.add(
        tf.assert_type(
            losses,
            self._dtype,
            message="losses must be the same dtype as predictions"))
    average_loss = tf.reduce_mean(losses)

    return average_loss, pre_train_ops, restart_ops
