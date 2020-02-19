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
combinations of `Term`s, and it contains some additional logic for combining
"compatible" `Term`s.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import helpers


class BasicExpression(helpers.RateObject):
  """Object representing a linear combination of `Term`s.

  The `BasicExpression` object is, along with `Expression`, one of the two core
  objects used to represent a quantity that can be minimized or constrained as
  part of a rate-constrained problem. It represents a linear combination of
  `Term`s. To this end, it supports negation, addition, subtraction, scalar
  multiplication and division.

  A new `BasicExpression` object is constructed from a list of `Term`s (which
  are summed). The reason for taking a list of `Term`s, instead of only a single
  `Term` representing the entire linear combination, is that, unlike `Tensor`s,
  two `Term`s can only be added or subtracted if they're "compatible" (which is
  a notion defined by the `Term` itself).
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
    for tt in source:
      key = tt.key
      if key in indices:
        self._terms[indices[key]] += tt
      else:
        indices[tt.key] = len(self._terms)
        self._terms.append(tt)

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
    for tt in source:
      key = tt.key
      if key in indices:
        self._terms[indices[key]] -= tt
      else:
        indices[tt.key] = len(self._terms)
        self._terms.append(-tt)

  def __init__(self, terms):
    """Creates a new `BasicExpression`.

    The reason for taking a collection of `Term`s, instead of only a single
    `Term` representing the entire linear combination, is that, unlike
    `Tensor`s, two `Term`s can only be added or subtracted if they're
    "compatible" (which is a notion defined by the `Term` itself).

    Args:
      terms: collection of `Term`s to sum in the `BasicExpression`.
    """
    # This object contains the "_terms" member variable, representing a linear
    # combination of `Term` objects. Like `Tensor`s, `Term`s support negation,
    # scalar multiplication and division without restriction. Unlike `Tensor`s,
    # however, only "compatible" `Term`s may be added or subtracted. Two `Term`s
    # are compatible iff they have the same key (returned by their "key"
    # method). When we add or subtract two `BasicExpression`s, compatible
    # `Term`s are added or subtracted, and incompatible `Term`s are put in their
    # own dict entries.
    #
    # We use a list to store the Terms (instead of e.g. a dict mapping keys to
    # Terms), so that we can preserve the order of the Terms (of course, this
    # isn't the only way to handle it). This is needed to support distributed
    # optimization, since it results in the DeferredVariables upon which the
    # Terms depend having a consistent order from machine-to-machine.
    self._terms = []
    self._add_terms(terms)

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
    if not isinstance(scalar, numbers.Number):
      raise TypeError("BasicExpression objects only support *scalar* "
                      "multiplication")
    return BasicExpression([tt * scalar for tt in self._terms])

  def __rmul__(self, scalar):
    """Returns the result of multiplying by a scalar."""
    return self.__mul__(scalar)

  def __truediv__(self, scalar):
    """Returns the result of dividing by a scalar."""
    if not isinstance(scalar, numbers.Number):
      raise TypeError("BasicExpression objects only support *scalar* division")
    return BasicExpression([tt / scalar for tt in self._terms])

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (BasicExpression / scalar) is allowed, but (scalar / BasicExpression) is
  # not.

  def __neg__(self):
    """Returns the result of negating this `BasicExpression`."""
    return BasicExpression([-tt for tt in self._terms])

  def __add__(self, other):
    """Returns the result of adding two `BasicExpression`s."""
    if not isinstance(other, BasicExpression):
      raise TypeError("BasicExpression objects can only be added to each "
                      "other")

    result = BasicExpression(self._terms)
    # pylint: disable=protected-access
    result._add_terms(other._terms)
    # pylint: enable=protected-access
    return result

  def __sub__(self, other):
    """Returns the result of subtracting two `BasicExpression`s."""
    if not isinstance(other, BasicExpression):
      raise TypeError("BasicExpression objects can only be subtracted from "
                      "each other")

    result = BasicExpression(self._terms)
    # pylint: disable=protected-access
    result._sub_terms(other._terms)
    # pylint: enable=protected-access
    return result

  def evaluate(self, memoizer):
    """Computes and returns the value of this `BasicExpression`.

    Args:
      memoizer: dict, which memoizes portions of the calculation to simplify the
        resulting TensorFlow graph. It must contain the keys
        "denominator_lower_bound" and "global_step", with the corresponding
        values being the minimum allowed value of a rate denominator (a float),
        and the current iterate (a non-negative integer, starting at zero),
        respectively.

    Returns:
      A `DeferredTensor` containing the value of this `BasicExpression`.
    """
    values = []
    for tt in self._terms:
      term_value = tt.evaluate(memoizer)
      values.append(term_value)

    # We create a list of values, and sum them all-at-once (instead of adding
    # them one-by-one inside the above loop) to limit how deeply the closures
    # inside the DeferredTensor will be nested.
    return deferred_tensor.DeferredTensor.apply(lambda *args: sum(args),
                                                *values)
