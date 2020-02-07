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

from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import helpers


class BasicExpression(helpers.RateObject):
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
    """Adds a collection of `Term`s to this `BasicExpression` (in place).

    Args:
      source: collection of `Term`s to add to this `BasicExpression`.
    """
    # We use a list to store the Terms (instead of e.g. a dict mapping keys to
    # Terms), so that we can preserve the order of the Terms. As a consequence,
    # we set up an indices dict every time we add two BasicExpressions (of
    # course, this isn't the only way to handle it).
    indices = {tt.key: ii for ii, tt in enumerate(self._terms)}
    for cc in source:
      key = cc.key
      if key in indices:
        self._terms[indices[key]] += cc
      else:
        indices[cc.key] = len(self._terms)
        self._terms.append(cc)

  def _sub_terms(self, source):
    """Subtracts a collection of `Term`s from this `BasicExpression` (in place).

    Args:
      source: collection of `Term`s to subtract from this `BasicExpression`.
    """
    # We use a list to store the Terms (instead of e.g. a dict mapping keys to
    # Terms), so that we can preserve the order of the Terms. As a consequence,
    # we set up an indices dict every time we subtract two BasicExpressions (of
    # course, this isn't the only way to handle it).
    indices = {tt.key: ii for ii, tt in enumerate(self._terms)}
    for cc in source:
      key = cc.key
      if key in indices:
        self._terms[indices[key]] -= cc
      else:
        indices[cc.key] = len(self._terms)
        self._terms.append(-cc)

  def __init__(self, terms, tensor=0.0):
    """Creates a new `BasicExpression`.

    The reason for taking a collection of `Term`s, instead of only a single
    `Term` representing the entire linear combination, is that, unlike
    `Tensor`s, two `Term`s can only be added or subtracted if they're
    "compatible" (which is a notion defined by the `Term` itself).

    Args:
      terms: collection of `Term`s to sum in the `BasicExpression`.
      tensor: optional scalar `DeferredTensor` or `Tensor`-like object to add to
        the sum of `Term`s.

    Raises:
      TypeError: if "tensor" is not a `DeferredTensor` or `Tensor`-like object.
    """
    # This object contains two member variables: "_terms", representing a linear
    # combination of Term objects, and "_tensor", representing an additional
    # Tensor-like object to include in the sum. The "_tensor" variable is
    # capable of representing a linear combination of Tensors, since Tensors
    # support negation, addition, subtraction, scalar multiplication and
    # division.
    #
    # It isn't so simple for Terms. Like Tensors, they support negation, scalar
    # multiplication and division without restriction. Unlike Tensors, however,
    # only "compatible" Terms may be added or subtracted. Two Terms are
    # compatible iff they have the same key (returned by their "key" method).
    # When we add or subtract two BasicExpressions, compatible Terms are added
    # or subtracted within the _terms list, and incompatible Terms are appended
    # to the list.
    #
    # We use a list to store the Terms (instead of e.g. a dict mapping keys to
    # Terms), so that we can preserve the order of the Terms (of course, this
    # isn't the only way to handle it). This is needed to support distributed
    # optimization, since it results in the DeferredVariables upon which the
    # Terms depend having a consistent order from machine-to-machine.
    self._terms = []
    self._add_terms(terms)
    if isinstance(tensor, deferred_tensor.DeferredTensor):
      self._tensor = tensor
    elif not isinstance(tensor, helpers.RateObject):
      self._tensor = deferred_tensor.DeferredTensor(tensor)
    else:
      raise TypeError("tensor argument to BasicExpression's constructor "
                      "should be a DeferredTensor or a Tensor-like object")

  @property
  def tensor(self):
    """Returns the `DeferredTensor` contained in this `BasicExpression`."""
    return self._tensor

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
    for tt in self._terms:
      result &= tt.is_differentiable
    return result

  def __mul__(self, scalar):
    """Returns the result of multiplying by a scalar."""
    # Ideally, we'd check that "scalar" is a scalar Tensor, or is a type that
    # can be converted to a scalar Tensor. Unfortunately, this includes a lot of
    # possible types, so the easiest solution would be to actually perform the
    # conversion, and then check that the resulting Tensor has only one element.
    # This, however, would add a dummy element to the Tensorflow graph, and
    # wouldn't work for a Tensor with an unknown size. Hence, we only check that
    # "scalar" is not a type that we know for certain is disallowed: an object
    # internal to this library.
    if isinstance(scalar, helpers.RateObject):
      raise TypeError("BasicExpression objects only support *scalar* "
                      "multiplication")
    terms = [tt * scalar for tt in self._terms]
    return BasicExpression(terms, self._tensor * scalar)

  def __rmul__(self, scalar):
    """Returns the result of multiplying by a scalar."""
    return self.__mul__(scalar)

  def __truediv__(self, scalar):
    """Returns the result of dividing by a scalar."""
    # See comment in __mul__.
    if isinstance(scalar, helpers.RateObject):
      raise TypeError("BasicExpression objects only support *scalar* division")
    terms = [tt / scalar for tt in self._terms]
    return BasicExpression(terms, self._tensor / scalar)

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (BasicExpression / scalar) is allowed, but (scalar / BasicExpression) is
  # not.

  def __neg__(self):
    """Returns the result of negating this `BasicExpression`."""
    terms = [-tt for tt in self._terms]
    return BasicExpression(terms, -self._tensor)

  def __add__(self, other):
    """Returns the result of adding two `BasicExpression`s."""
    if isinstance(other, BasicExpression):
      result = BasicExpression(self._terms, self._tensor + other.tensor)
      # pylint: disable=protected-access
      result._add_terms(other._terms)
      # pylint: enable=protected-access
      return result
    elif not isinstance(other, helpers.RateObject):
      return BasicExpression(self._terms, self._tensor + other)
    else:
      raise TypeError("BasicExpression objects can only be added to each "
                      "other, or scalars")

  def __radd__(self, other):
    """Returns the result of adding two `BasicExpression`s."""
    return self.__add__(other)

  def __sub__(self, other):
    """Returns the result of subtracting two `BasicExpression`s."""
    if isinstance(other, BasicExpression):
      result = BasicExpression(self._terms, self._tensor - other.tensor)
      # pylint: disable=protected-access
      result._sub_terms(other._terms)
      # pylint: enable=protected-access
      return result
    elif not isinstance(other, helpers.RateObject):
      return BasicExpression(self._terms, self._tensor - other)
    else:
      raise TypeError("BasicExpression objects can only be subtracted from "
                      "each other, or scalars")

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
        Returns A (`DeferredTensor`, list) pair containing (i) the value of this
        `BasicExpression`, and (ii) a list of `DeferredVariable`s containing the
        internal state upon which the `BasicExpression` evaluation depends.

    Returns:
      A (`DeferredTensor`, list) pair containing (i) the value of this
      `BasicExpression`, and (ii) a list of `DeferredVariable`s containing the
      internal state upon which the `BasicExpression` evaluation depends.
    """
    values = [self._tensor]
    variables = deferred_tensor.DeferredVariableList()
    for tt in self._terms:
      term_value, term_variables = tt.evaluate(memoizer)
      values.append(term_value)
      variables += term_variables

    # We create a list of values, and sum them all-at-once (instead of adding
    # them one-by-one inside the above loop) to limit how deeply the closures
    # inside the DeferredTensor will be nested.
    return deferred_tensor.DeferredTensor.apply(lambda *args: sum(args),
                                                *values), variables.list
