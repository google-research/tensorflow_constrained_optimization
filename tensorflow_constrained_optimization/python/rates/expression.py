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
"""Contains the `Expression` class.

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
formulation). In other words, an `Expression` needs to know not only exactly
what quantity it represents (the "constraints" portion), but also a
{sub,super}differentiable approximation of it (the "penalty" portion).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numbers
import six

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import constraint
from tensorflow_constrained_optimization.python.rates import helpers
from tensorflow_constrained_optimization.python.rates import term


@six.add_metaclass(abc.ABCMeta)
class Expression(helpers.RateObject):
  """Represents an expression that can be penalized or constrained.

  An `Expression`, like a `BasicExpression`, represents a linear combination of
  terms. However, it actually includes *two* `BasicExpression`s, one of
  which--the "penalty" portion--is used when the expression is being minimized
  (in the objective function) or penalized (to satisfy a constraint), and the
  second of which--the "constraint" portion--is used when the expression is
  being constrained.

  Typically, the "penalty" and "constraint" portions will be different
  approximations of the same quantity, e.g. the latter could be a zero-one-based
  rate (say, a false positive rate), and the former a hinge approximation of it.

  In addition, an `Expression` can (optionally) contain a set of associated
  constraints. Technically, these can be anything: they're simply additional
  constraints, which may or may not have anything to do with the `Expression` to
  which they're attached. In practice, they'll usually represent conditions that
  are required for the associated `Expression` to make sense. For example, we
  could construct an `Expression` representing "the false positive rate of f(x)
  at the threshold for which the true positive rate is at least 0.9" with the
  expression being "the false positive rate of f(x) - t", and the extra
  constraint being "the true positive rate of f(x) - t is at least 0.9", where
  "t" is an implicit threshold. These extra constraints will ultimately be
  included in any optimization problem that includes the associated `Expression`
  (or an `Expression` derived from it).
  """

  @abc.abstractproperty
  def penalty_expression(self):
    """Returns the `BasicExpression` used for the "penalty" portion.

    An `Expression` contains *two* `BasicExpression`s, one of which--the
    "penalty" portion--is used when the expression is being minimized (in the
    objective function) or penalized (to satisfy a constraint), while the
    second--the "constraint" portion--is used when the expression is being
    constrained.
    """

  @abc.abstractproperty
  def constraint_expression(self):
    """Returns the `BasicExpression` used for the "constraint" portion.

    An `Expression` contains *two* `BasicExpression`s, one of which--the
    "penalty" portion--is used when the expression is being minimized (in the
    objective function) or penalized (to satisfy a constraint), while the
    second--the "constraint" portion--is used when the expression is being
    constrained.
    """

  @abc.abstractproperty
  def extra_constraints(self):
    """Returns the list of extra `Constraint`s."""

  @abc.abstractmethod
  def __mul__(self, scalar):
    """Returns the result of multiplying by a scalar."""

  def __rmul__(self, scalar):
    """Returns the result of multiplying by a scalar."""
    return self.__mul__(scalar)

  @abc.abstractmethod
  def __truediv__(self, scalar):
    """Returns the result of dividing by a scalar."""

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (Expression / scalar) is allowed, but (scalar / Expression) is not.

  @abc.abstractmethod
  def __neg__(self):
    """Returns the result of negating this `Expression`."""

  def __add__(self, other):
    """Returns the result of adding two `Expression`s."""
    if not isinstance(other, helpers.RateObject):
      # BasicExpressions do not support scalar addition, so we first need to
      # convert the scalar into an Expression.
      other_basic_expression = basic_expression.BasicExpression(
          [term.TensorTerm(other)])
      other = ExplicitExpression(other_basic_expression, other_basic_expression)
    elif not isinstance(other, Expression):
      raise TypeError("Expression objects can only be added to each other, or "
                      "scalars")

    return SumExpression([self, other])

  def __radd__(self, other):
    """Returns the result of adding two `Expression`s."""
    return self.__add__(other)

  def __sub__(self, other):
    """Returns the result of subtracting two `Expression`s."""
    return self.__add__(other.__neg__())

  def __rsub__(self, other):
    """Returns the result of subtracting two `Expression`s."""
    return self.__neg__().__add__(other)

  def __le__(self, other):
    """Returns a `Constraint` representing self <= other."""
    return constraint.Constraint(self - other)

  def __ge__(self, other):
    """Returns a `Constraint` representing self >= other."""
    return constraint.Constraint(other - self)

  # We don't make __eq__ create an equality constraint for two reasons:
  #   1) We might want to put Expressions into sets or something.
  #   2) We probably won't be able to optimize subject to equality constraints
  #      very effectively, anyway.


class ExplicitExpression(Expression):
  """Represents explicit penalty and constraint `BasicExpressions`."""

  def __init__(self, penalty_expression, constraint_expression):
    """Creates a new `ExplicitExpression`.

    An `Expression` represents a quantity that will be minimized/maximized or
    constrained. Internally, it's actually represented as *two*
    `BasicExpression`s, one of which--the "penalty" portion--is used when the
    expression is being minimized (in the objective function) or penalized (to
    satisfy a constraint), and the second of which--the "constraint" portion--is
    used when the expression is being constrained. These two `BasicExpression`s
    are the two parameters of this function.

    Args:
      penalty_expression: `BasicExpression` that will be used for the "penalty"
        portion of the optimization (i.e. when optimizing the model parameters).
        It should be {sub,semi}differentiable.
      constraint_expression: `BasicExpression` that will be used for the
        "constraint" portion of the optimization (i.e. when optimizing the
        constraints). It does not need to be {sub,semi}differentiable.
    """
    self._penalty_expression = penalty_expression
    self._constraint_expression = constraint_expression

  @property
  def penalty_expression(self):
    return self._penalty_expression

  @property
  def constraint_expression(self):
    return self._constraint_expression

  @property
  def extra_constraints(self):
    return []

  def __mul__(self, scalar):
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Expression objects only support *scalar* multiplication")
    return ExplicitExpression(self._penalty_expression * scalar,
                              self._constraint_expression * scalar)

  def __truediv__(self, scalar):
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Expression objects only support *scalar* division")
    return ExplicitExpression(self._penalty_expression / scalar,
                              self._constraint_expression / scalar)

  def __neg__(self):
    return ExplicitExpression(-self._penalty_expression,
                              -self._constraint_expression)


class ConstrainedExpression(Expression):
  """Represents a list of constraints wrapped around an `Expression`."""

  def __init__(self, expression, extra_constraints):
    """Creates a new `ConstrainedExpression`.

    The "extra_constraints" parameter is used to specify additional constraints
    that should be added to any optimization problem involving this given
    "expression". Technically, these can be anything: they're simply additional
    constraints, which may or may not have anything to do with the `Expression`
    to which they're attached. In practice, they'll usually represent conditions
    that are required for the associated `Expression` to make sense. For
    example, we could construct an `Expression` representing "the false positive
    rate of f(x) at the threshold for which the true positive rate is at least
    0.9" with the expression being "the false positive rate of f(x) - t", and
    the extra constraint being "the true positive rate of f(x) - t is at least
    0.9", where "t" is an implicit threshold. These extra constraints will
    ultimately be included in any optimization problem that includes the
    associated `Expression` (or an `Expression` derived from it).

    Args:
      expression: `Expression` around which the given constraint dependencies
        will be wrapped.
      extra_constraints: optional collection of `Constraint`s required by this
        `Expression`.
    """
    self._expression = expression
    self._extra_constraints = constraint.ConstraintList(extra_constraints)

  @property
  def penalty_expression(self):
    return self._expression.penalty_expression

  @property
  def constraint_expression(self):
    return self._expression.constraint_expression

  @property
  def extra_constraints(self):
    # The "list" property of a ConstraintList returns a copy.
    return self._extra_constraints.list

  def __mul__(self, scalar):
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Expression objects only support *scalar* multiplication")
    return ConstrainedExpression(self._expression * scalar,
                                 self._extra_constraints)

  def __truediv__(self, scalar):
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Expression objects only support *scalar* division")
    return ConstrainedExpression(self._expression / scalar,
                                 self._extra_constraints)

  def __neg__(self):
    return ConstrainedExpression(-self._expression, self._extra_constraints)


class SumExpression(Expression):
  """Represents a sum of `Expression`s."""

  @classmethod
  def _expression_to_list(cls, expression):
    """Flattens an `Expression` by recursing into `SumExpression`s."""
    if isinstance(expression, SumExpression):
      result = []
      for subexpression in expression._expressions:  # pylint: disable=protected-access
        result += cls._expression_to_list(subexpression)
    else:
      result = [expression]
    return result

  def __init__(self, expressions):
    """Creates a new `SumExpression`.

    In the `Expression` base class, unary operators are abstract, but binary
    plus and minus are defined so as to construct `SumExpression`s from the
    added or subtracted `Expression`s. It could, of course, simply add or
    subtract the contained `BasicExpression`s, and construct a new
    `ExplicitExpression` from them, but keeping each component of an
    `Expression` separate allows us to create additional implementations of
    `Expression` that do some extra processing before being evaluated.

    Args:
      expressions: collection of `Expression`s that this object should sum.
    """
    self._expressions = []
    for subexpression in expressions:
      self._expressions += SumExpression._expression_to_list(subexpression)

  @property
  def penalty_expression(self):
    # Notice that this summation will perform some simplification, since
    # BasicExpressions combine compatible Terms when added.
    result = basic_expression.BasicExpression([])
    for subexpression in self._expressions:
      result += subexpression.penalty_expression
    return result

  @property
  def constraint_expression(self):
    # Notice that this summation will perform some simplification, since
    # BasicExpressions combine compatible Terms when added.
    result = basic_expression.BasicExpression([])
    for subexpression in self._expressions:
      result += subexpression.constraint_expression
    return result

  @property
  def extra_constraints(self):
    result = constraint.ConstraintList()
    for subexpression in self._expressions:
      result += subexpression.extra_constraints
    # The "list" property of a ConstraintList returns a copy.
    return result.list

  def __mul__(self, scalar):
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Expression objects only support *scalar* multiplication")
    return SumExpression(
        [subexpression * scalar for subexpression in self._expressions])

  def __truediv__(self, scalar):
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Expression objects only support *scalar* division")
    return SumExpression(
        [subexpression / scalar for subexpression in self._expressions])

  def __neg__(self):
    return SumExpression(
        [-subexpression for subexpression in self._expressions])


class BoundedExpression(Expression):
  """Represents a pair of upper and lower bound `Expressions`."""

  def __init__(self, lower_bound, upper_bound, scalar=1.0):
    """Creates a new `BoundedExpression`.

    Sometimes, the quantity that we wish to optimize or constrain cannot be
    represented directly, but *can* be bounded. In these cases, a
    `BoundedExpression` can be used. It contains both an upper and lower bound,
    and chooses which to use based on the sign of the `BoundedExpression` (i.e.
    so that we'll maximize a lower bound and minimize an upper bound, instead of
    the other way around).

    If we only have a one-sided bound, an `InvalidExpression`, which raises an
    error if it's used, can be placed in the other side, to ensure that we don't
    try to use it.

    Args:
      lower_bound: `Expression` representing a lower bound on the quantity of
        interest.
      upper_bound: `Expression` representing an upper bound on the quantity of
        interest.
      scalar: float, a quantity by which to multiply this `BoundedExpression`.
    """
    self._lower_bound = lower_bound
    self._upper_bound = upper_bound
    self._scalar = scalar

  @property
  def penalty_expression(self):
    # When this is called, we will always be extracting the penalty expression
    # for minimization or upper-bounding. Hence, we'll use the lower bound if
    # the scalar is negative, and the upper bound otherwise.
    #
    # Notice that we scale the lower bound or upper bound Expression before
    # extracting the BasicExpression, instead of extracting the BasicExpression
    # and scaling it. The reason for this is that further BoundedExpressions
    # could be nested inside this one, and we need to be sure that their scalars
    # are up-to-date.
    if self._scalar < 0:
      return (self._lower_bound * self._scalar).penalty_expression
    elif self._scalar > 0:
      return (self._upper_bound * self._scalar).penalty_expression
    return basic_expression.BasicExpression([])

  @property
  def constraint_expression(self):
    # See comment in penalty_expression.
    if self._scalar < 0:
      return (self._lower_bound * self._scalar).constraint_expression
    elif self._scalar > 0:
      return (self._upper_bound * self._scalar).constraint_expression
    return basic_expression.BasicExpression([])

  @property
  def extra_constraints(self):
    # See comment in penalty_expression.
    if self._scalar < 0:
      return (self._lower_bound * self._scalar).extra_constraints
    elif self._scalar > 0:
      return (self._upper_bound * self._scalar).extra_constraints
    return []

  def __mul__(self, scalar):
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Expression objects only support *scalar* multiplication")
    return BoundedExpression(self._lower_bound, self._upper_bound,
                             self._scalar * scalar)

  def __truediv__(self, scalar):
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Expression objects only support *scalar* division")
    return BoundedExpression(self._lower_bound, self._upper_bound,
                             self._scalar / scalar)

  def __neg__(self):
    return BoundedExpression(self._lower_bound, self._upper_bound,
                             -self._scalar)


class InvalidExpression(Expression):
  """Represents an invalid `Expression`."""

  def __init__(self, message):
    """Creates a new `InvalidExpression`.

    While a `BoundedExpression` usually contains both a lower bound an an upper
    bound, it could contain only one of the two, in which case the other should
    be an `InvalidExpression`. This makes it so that if someone tries to
    maximize an upper bound, or minimize a lower bound, an error will be raised.

    Args:
      message: string to raise when "penalty_expression",
        "constraint_expression" or "extra_constraints" is called.
    """
    self._message = message

  @property
  def penalty_expression(self):
    raise RuntimeError(self._message)

  @property
  def constraint_expression(self):
    raise RuntimeError(self._message)

  @property
  def extra_constraints(self):
    raise RuntimeError(self._message)

  def __mul__(self, scalar):
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Expression objects only support *scalar* multiplication")
    return self

  def __truediv__(self, scalar):
    if not isinstance(scalar, numbers.Number):
      raise TypeError("Expression objects only support *scalar* division")
    return self

  def __neg__(self):
    return self
