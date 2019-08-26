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
"""Contains the `Expression` and `Constraint` classes.

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

from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import helpers


class Constraint(helpers.RateObject):
  """Represents an inequality constraint.

  This class is nothing but a thin wrapper around an `Expression`, and
  represents the constraint that the wrapped expression is non-positive.
  """

  def __init__(self, expression):
    """Constructs a new `Constraint` object from an `Expression`."""
    self._expression = expression

  @property
  def expression(self):
    """Returns the `Expression` that is constrained to be nonpositive."""
    return self._expression


class Expression(helpers.RateObject):
  """Represents an expression that can be penalized or constrained.

  An `Expression`, like a `BasicExpression`, represents a linear combination of
  `Term`s and `Tensor`s. Internally, it's actually represented as *two*
  `BasicExpression`s, one of which--the "penalty" portion--is used when the
  expression is being minimized (in the objective function) or penalized (to
  satisfy a constraint), and the second of which--the "constraint" portion--is
  used when the expression is being constrained.

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

  def __init__(self,
               penalty_expression,
               constraint_expression,
               extra_variables=None,
               extra_constraints=None):
    """Creates a new `Expression`.

    An `Expression` represents a quantity that will be minimized/maximized or
    constrained. Internally, it's actually represented as *two*
    `BasicExpression`s, one of which--the "penalty" portion--is used when the
    expression is being minimized (in the objective function) or penalized (to
    satisfy a constraint), and the second of which--the "constraint" portion--is
    used when the expression is being constrained. These two `BasicExpression`s
    are the first two parameters of this function.

    The third parameter--"extra_variables"--should contain any
    `DeferredVariable`s that are owned by this `Expression`. This is most
    commonly used for slack variables.

    The fourth parameter--"extra_constraints"--is used to specify additional
    constraints that should be added to any optimization problem involving this
    `Expression`. Technically, these can be anything: they're simply additional
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
      penalty_expression: `BasicExpression` that will be used for the "penalty"
        portion of the optimization (i.e. when optimizing the model parameters).
        It should be {sub,semi}differentiable.
      constraint_expression: `BasicExpression` that will be used for the
        "constraint" portion of the optimization (i.e. when optimizing the
        constraints). It does not need to be {sub,semi}differentiable.
      extra_variables: collection of `DeferredVariable`s owned by this
        `Expression`.
      extra_constraints: collection of `Constraint`s required by this
        `Expression`.
    """
    self._penalty_expression = penalty_expression
    self._constraint_expression = constraint_expression

    if extra_variables is None:
      self._extra_variables = set()
    else:
      self._extra_variables = set(extra_variables)
    if not all(
        isinstance(extra_variable, deferred_tensor.DeferredVariable)
        for extra_variable in self._extra_variables):
      raise TypeError("all elements of extra_variables must be "
                      "DeferredVariable objects")

    if extra_constraints is None:
      self._extra_constraints = set()
    else:
      self._extra_constraints = set(extra_constraints)
    if not all(
        isinstance(extra_constraint, Constraint)
        for extra_constraint in self._extra_constraints):
      raise TypeError("all elements of extra_constraints must be Constraint "
                      "objects")

  @property
  def penalty_expression(self):
    """Returns the `BasicExpression` used for the "penalty" portion.

    An `Expression` contains *two* `BasicExpression`s, one of which--the
    "penalty" portion--is used when the expression is being minimized (in the
    objective function) or penalized (to satisfy a constraint), while the
    second--the "constraint" portion--is used when the expression is being
    constrained.
    """
    return self._penalty_expression

  @property
  def constraint_expression(self):
    """Returns the `BasicExpression` used for the "constraint" portion.

    An `Expression` contains *two* `BasicExpression`s, one of which--the
    "penalty" portion--is used when the expression is being minimized (in the
    objective function) or penalized (to satisfy a constraint), while the
    second--the "constraint" portion--is used when the expression is being
    constrained.
    """
    return self._constraint_expression

  @property
  def extra_variables(self):
    """Returns the set of extra `DeferredVariable`s."""
    return self._extra_variables

  @property
  def extra_constraints(self):
    """Returns the set of extra `Constraint`s."""
    return self._extra_constraints

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
      raise TypeError("Expression objects only support *scalar* multiplication")
    return Expression(
        self._penalty_expression * scalar,
        self._constraint_expression * scalar,
        extra_variables=self._extra_variables,
        extra_constraints=self._extra_constraints)

  def __rmul__(self, scalar):
    """Returns the result of multiplying by a scalar."""
    return self.__mul__(scalar)

  def __truediv__(self, scalar):
    """Returns the result of dividing by a scalar."""
    # See comment in __mul__.
    if isinstance(scalar, helpers.RateObject):
      raise TypeError("Expression objects only support *scalar* division")
    return Expression(
        self._penalty_expression / scalar,
        self._constraint_expression / scalar,
        extra_variables=self._extra_variables,
        extra_constraints=self._extra_constraints)

  # __rtruediv__ is not implemented since we only allow *scalar* division, i.e.
  # (Expression / scalar) is allowed, but (scalar / Expression) is not.

  def __neg__(self):
    """Returns the result of negating this `Expression`."""
    # We don't negate extra_constraints since these constraints are not actually
    # a part of the expression itself: they're merely associated with it.
    # Typically, they're used to impose definitional properties of the
    # expression. For example, using the example in the class docstring:
    #
    #   "The false positive rate at the threshold for which the true positive
    #   rate is at least 90%"
    #
    # The expression is "the false positive rate at threshold t", and the
    # constraint is "at threshold t, the true positive rate is at least 90%".
    # Negation should yield:
    #
    #   "The negative false positive rate at the threshold for which the true
    #   positive rate is at least 90%"
    #
    # In other words, the constraint should not change. If we were to also
    # negate the constraint, we'd get:
    #
    #   "The negative false positive rate at the threshold for which the true
    #   positive rate is at most 90%"
    #
    # But this wouldn't make sense.
    return Expression(
        -self._penalty_expression,
        -self._constraint_expression,
        extra_variables=self._extra_variables,
        extra_constraints=self._extra_constraints)

  def __add__(self, other):
    """Returns the result of adding two `Expression`s."""
    if isinstance(other, Expression):
      return Expression(
          self._penalty_expression + other.penalty_expression,
          self._constraint_expression + other.constraint_expression,
          extra_variables=self._extra_variables | other.extra_variables,
          extra_constraints=self._extra_constraints | other.extra_constraints)
    elif not isinstance(other, helpers.RateObject):
      return Expression(
          self._penalty_expression + other,
          self._constraint_expression + other,
          extra_variables=self._extra_variables,
          extra_constraints=self._extra_constraints)
    else:
      raise TypeError("Expression objects can only be added to each other, "
                      "or scalars")

  def __radd__(self, other):
    """Returns the result of adding two `Expression`s."""
    return self.__add__(other)

  def __sub__(self, other):
    """Returns the result of subtracting two `Expression`s."""
    if isinstance(other, Expression):
      return Expression(
          self._penalty_expression - other.penalty_expression,
          self._constraint_expression - other.constraint_expression,
          extra_variables=self._extra_variables | other.extra_variables,
          extra_constraints=self._extra_constraints | other.extra_constraints)
    elif not isinstance(other, helpers.RateObject):
      return Expression(
          self._penalty_expression - other,
          self._constraint_expression - other,
          extra_variables=self._extra_variables,
          extra_constraints=self._extra_constraints)
    else:
      raise TypeError("Expression objects can only be subtracted from each "
                      "other, or scalars")

  def __rsub__(self, other):
    """Returns the result of subtracting two `Expression`s."""
    # We assert if "other" is an Expression, since if it was, then __sub__ would
    # have been called instead of __rsub__.
    assert not isinstance(other, Expression)
    if not isinstance(other, helpers.RateObject):
      return Expression(
          other - self._penalty_expression,
          other - self._constraint_expression,
          extra_variables=self._extra_variables,
          extra_constraints=self._extra_constraints)
    else:
      raise TypeError("Expression objects can only be subtracted from each "
                      "other, or scalars")

  def __le__(self, other):
    """Returns a `Constraint` representing self <= other."""
    return Constraint(self - other)

  def __ge__(self, other):
    """Returns a `Constraint` representing self >= other."""
    return Constraint(other - self)

  # We don't make __eq__ create an equality constraint for two reasons:
  #   1) We might want to put Expressions into sets or something.
  #   2) We probably won't be able to optimize subject to equality constraints
  #      very effectively, anyway.
