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
"""Contains the `BasicExpression` class.

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
set of negatively-labeled examples (the false positive rate).

The above description is somewhat oversimplified. Most relevantly here, an
`Expression` object actually consists of two `BasicExpression` objects: one for
the "penalty" portion of the problem (which is used when optimizing the model
parameters), and the other for the "constraint" portion (used when optimizing
the constraints, e.g. the Lagrange multipliers, if using the Lagrangian
formulation). It is the `BasicExpression` object that actually represents linear
combinations of `Term`s and `Tensor`s, and it contains some additional logic for
combining "compatible" `Term`s.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BasicExpression(object):
  """Object representing a linear combination of `Term`s and `Tensor`s.

  The `BasicExpression` object is, along with `Expression`, one of the two core
  objects used to represent a quantity that can be minimized or constrained as
  part of a rate-constrained problem. It represents a linear combination of
  `Term`s and `Tensor`s. To this end, it supports negation, addition,
  subtraction, scalar multiplication and division.

  A new `BasicExpression` object is constructed from a list of `Term`s (which
  are summed), and a single `Tensor`. The reason for taking a list of `Term`s,
  instead of only a single `Term` representing the entire linear combination, is
  that, unlike `Tensor`s, two `Term`s can only be added or subtracted if they're
  "compatible" (which is a notion defined by the `Term` itself).
  """

  def _add_terms(self, source):
    """Adds a list of `Term`s to this `BasicExpression` (in place).

    Args:
      source: list of `Term`s to add to this `BasicExpression`.
    """
    for tt in source:
      key = tt.key
      if key in self._terms:
        self._terms[key] += tt
      else:
        self._terms[key] = tt

  def _sub_terms(self, source):
    """Subtracts a list of `Term`s from this `BasicExpression` (in place).

    Args:
      source: list of `Term`s to subtract from this `BasicExpression`.
    """
    for tt in source:
      key = tt.key
      if key in self._terms:
        self._terms[key] -= tt
      else:
        self._terms[key] = -tt

  def __init__(self, terms, tensor=0.0):
    """Creates a new `BasicExpression`.

    The reason for taking a list of `Term`s, instead of only a single `Term`
    representing the entire linear combination, is that, unlike `Tensor`s, two
    `Term`s can only be added or subtracted if they're "compatible" (which is a
    notion defined by the `Term` itself).

    Args:
      terms: list of `Term`s to sum in the `BasicExpression`.
      tensor: optional scalar `Tensor` to add to the sum of `Term`s.
    """
    # This object contains two member variables: "_terms", representing a linear
    # combination of `Term` objects, and "_tensor", representing an additional
    # `Tensor` object to include in the sum. The "_tensor" variable is capable
    # of representing a linear combination of `Tensor`s, since `Tensor`s support
    # negation, addition, subtraction, scalar multiplication and division.
    #
    # It isn't so simple for `Term`s. Like `Tensor`s, they support negation,
    # scalar multiplication and division without restriction. Unlike `Tensor`s,
    # however, only "compatible" `Term`s may be added or subtracted. Two `Term`s
    # are compatible iff they have the same key (returned by their "key"
    # method). For this reason, "_terms" is a dict mapping keys to `Term`s.
    # When we add or subtract two `BasicExpression`s, compatible `Term`s are
    # added or subtracted, and incompatible `Term`s are put in their own dict
    # entries.
    self._terms = {}
    self._add_terms(terms)
    self._tensor = tensor

  @property
  def terms(self):
    """Returns the list of `Term`s contained in this `BasicExpression`."""
    # Notice that we only return the values in the _terms dict. The keys are
    # only used for combining common terms (in the binary operators).
    return self._terms.values()

  @property
  def tensor(self):
    """Returns the `Tensor` contained in this `BasicExpression`."""
    return self._tensor

  @property
  def dtype(self):
    """Returns the `tf.DType` of this `BasicExpression`.

    Returns:
      The dtype of this `BasicExpression`, i.e. the dtype of the `Tensor` that
      will be returned by evaluate().

    Raises:
      TypeError: if we are unable to determine the dtype (most likely because
        the `BasicExpression` contains terms with different dtypes).
    """
    result = None

    # If the tensor is actually a Tensor (and not a python float), then its
    # dtype must match those of the Terms.
    if tf.contrib.framework.is_tensor(self._tensor):
      result = self._tensor.dtype.base_dtype

    # Make sure that all terms have the same dtype.
    for tt in self._terms.values():
      tt_dtype = tt.dtype
      if result is None:
        result = tt_dtype
      elif tt_dtype != result:
        raise TypeError("all terms in an expression must have the same dtype")

    # Raise an error if the entire BasicExpression is just a python float (which
    # would be completely useless, since it would be a constant, hence couldn't
    # be minimized or constrained).
    if result is None:
      raise TypeError("unable to determine dtype of expression")

    return result

  @property
  def is_differentiable(self):
    """Returns true only if the losses are all differentiable.

    This property is used to check that non-differentiable losses (e.g. the
    zero-one loss) aren't used in contexts in which we will try to optimize over
    them. Non-differentiable losses, however, can be *evaluated* safely.

    Returns:
      True if all `Term`'s in this `BasicExpression` have
      {sub,super}differentiable losses. False otherwise.
    """
    result = True
    for tt in self._terms.values():
      result &= tt.is_differentiable
    return result

  def __mul__(self, scalar):
    """Returns the result of multiplying by a scalar."""
    # Ideally, we'd check that "scalar" is a scalar Tensor, or is a type that
    # can be converted to a scalar Tensor. Unfortunately, this includes a lot of
    # possible types, so the easiest solution would be to actually perform the
    # conversion, and then check that the resulting Tensor has only one element.
    # This, however, would add a dummy element to the Tensorflow graph, and
    # wouldn't work for a Tensor with an unknown size. Hence, we check only the
    # most common failure case (multiplication of two BasicExpressions).
    if isinstance(scalar, BasicExpression):
      raise TypeError(
          "BasicExpression objects only support *scalar* "
          "multiplication: you cannot multiply two BasicExpressions")
    terms = [tt * scalar for tt in self._terms.values()]
    return BasicExpression(terms, self._tensor * scalar)

  def __rmul__(self, scalar):
    """Returns the result of multiplying by a scalar."""
    return self.__mul__(scalar)

  def __truediv__(self, scalar):
    """Returns the result of dividing by a scalar."""
    # We check that "scalar" is not a BasicExpression, instead of checking that
    # it is a scalar, for the same reason as in __mul__.
    if isinstance(scalar, BasicExpression):
      raise TypeError("BasicExpression objects only support *scalar* "
                      "division: you cannot divide two BasicExpressions")
    terms = [tt / scalar for tt in self._terms.values()]
    return BasicExpression(terms, self._tensor / scalar)

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (BasicExpression / scalar) is allowed, but (scalar / BasicExpression) is
  # not.

  def __neg__(self):
    """Returns the result of negating this `BasicExpression`."""
    terms = [-tt for tt in self._terms.values()]
    return BasicExpression(terms, -self._tensor)

  def __add__(self, other):
    """Returns the result of adding two `BasicExpression`s."""
    if isinstance(other, BasicExpression):
      result = BasicExpression(self._terms.values(),
                               self._tensor + other.tensor)
      # pylint: disable=protected-access
      result._add_terms(other.terms)
      # pylint: enable=protected-access
      return result
    else:
      return BasicExpression(self._terms.values(), self._tensor + other)

  def __radd__(self, other):
    """Returns the result of adding two `BasicExpression`s."""
    return self.__add__(other)

  def __sub__(self, other):
    """Returns the result of subtracting two `BasicExpression`s."""
    if isinstance(other, BasicExpression):
      result = BasicExpression(self._terms.values(),
                               self._tensor - other.tensor)
      # pylint: disable=protected-access
      result._sub_terms(other.terms)
      # pylint: enable=protected-access
      return result
    else:
      return BasicExpression(self._terms.values(), self._tensor - other)

  def __rsub__(self, other):
    """Returns the result of subtracting two `BasicExpression`s."""
    return self.__neg__().__add__(other)

  def evaluate(self, memoizer):
    """Computes and returns the value of this `BasicExpression`.

    Args:
      memoizer: dict, which memoizes portions of the calculation to simplify the
        resulting TensorFlow graph. It must contain the keys
        "denominator_lower_bound" and "global_step", with the corresponding
        values being the minimum allowed value of a rate denominator (a python
        float), and the current iterate (starting at zero), respectively.

    Returns:
      A (`Tensor`, set, set) tuple containing the value of this
      `BasicExpression`, a set of `Operation`s that should be executed before
      each training step (to update the internal state upon which the
      `BasicExpression` evaluation depends), and a set of `Operation`s that can
      be executed to re-initialize this state.
    """
    value = self._tensor
    pre_train_ops = set()
    restart_ops = set()
    for tt in self._terms.values():
      tt_value, tt_pre_train_ops, tt_restart_ops = tt.evaluate(memoizer)
      value += tt_value
      pre_train_ops.update(tt_pre_train_ops)
      restart_ops.update(tt_restart_ops)
    # The value should already be the correct dtype--this cast just makes extra
    # sure.
    return tf.cast(value, dtype=self.dtype), pre_train_ops, restart_ops
