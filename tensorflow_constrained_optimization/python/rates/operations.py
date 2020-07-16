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
"""Functions for manipulating rate `Expression`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import expression
from tensorflow_constrained_optimization.python.rates import helpers
from tensorflow_constrained_optimization.python.rates import term


def wrap_rate(penalty_tensor, constraint_tensor=None):
  """Creates an `Expression` representing the given `Tensor`(s).

  The reason an `Expression` contains two `BasicExpression`s is that the
  "penalty" `BasicExpression` will be differentiable, while the "constraint"
  `BasicExpression` need not be. During optimization, the former will be used
  whenever we need to take gradients, and the latter otherwise.

  Args:
    penalty_tensor: scalar `Tensor`, the quantity to store in the "penalty"
      portion of the result (and also the "constraint" portion, if
      constraint_tensor is not provided).
    constraint_tensor: scalar `Tensor`, the quantity to store in the
      "constraint" portion of the result.

  Returns:
    An `Expression` wrapping the given `Tensor`(s).

  Raises:
    TypeError: if wrap_rate() is called on an `Expression`.
  """
  # Ideally, we'd check that "penalty_tensor" and "constraint_tensor" are scalar
  # Tensors, or are types that can be converted to a scalar Tensor.
  # Unfortunately, this includes a lot of possible types, so the easiest
  # solution would be to actually perform the conversion, and then check that
  # the resulting Tensor has only one element. This, however, would add a dummy
  # element to the Tensorflow graph, and wouldn't work for a Tensor with an
  # unknown size. Hence, we only check that "penalty_tensor" and
  # "constraint_tensor" are not types that we know for certain are disallowed:
  # objects internal to this library.
  if (isinstance(penalty_tensor, helpers.RateObject) or
      isinstance(constraint_tensor, helpers.RateObject)):
    raise TypeError("you cannot wrap an object that has already been wrapped")

  penalty_basic_expression = basic_expression.BasicExpression(
      [term.TensorTerm(deferred_tensor.ExplicitDeferredTensor(penalty_tensor))])
  if constraint_tensor is None:
    constraint_basic_expression = penalty_basic_expression
  else:
    constraint_basic_expression = basic_expression.BasicExpression([
        term.TensorTerm(
            deferred_tensor.ExplicitDeferredTensor(constraint_tensor))
    ])
  return expression.ExplicitExpression(penalty_basic_expression,
                                       constraint_basic_expression)


def upper_bound(expressions):
  """Creates an `Expression` upper bounding the given expressions.

  This function introduces a slack variable, and adds constraints forcing this
  variable to upper bound all elements of the given expression list. It then
  returns the slack variable.

  If you're going to be upper-bounding or minimizing the result of this
  function, then you can think of it as taking the `max` of its arguments. You
  should *never* lower-bound or maximize the result, however, since the
  consequence would be to increase the value of the slack variable, without
  affecting the contents of the expressions list.

  Args:
    expressions: list of `Expression`s, the quantities to upper-bound.

  Returns:
    An `Expression` representing an upper bound on the given expressions.

  Raises:
    ValueError: if the expressions list is empty.
    TypeError: if the expressions list contains a non-`Expression`.
  """
  if not expressions:
    raise ValueError("upper_bound cannot be given an empty expression list")
  if not all(isinstance(ee, expression.Expression) for ee in expressions):
    raise TypeError(
        "upper_bound expects a list of rate Expressions (perhaps you need to "
        "call wrap_rate() to create an Expression from a Tensor?)")

  # Ideally the slack variable would have the same dtype as the predictions, but
  # we might not know their dtype (e.g. in eager mode), so instead we always use
  # float32 with auto_cast=True.
  bound = deferred_tensor.DeferredVariable(
      0.0,
      trainable=True,
      name="tfco_upper_bound",
      dtype=tf.float32,
      auto_cast=True)

  bound_basic_expression = basic_expression.BasicExpression(
      [term.TensorTerm(bound)])
  bound_expression = expression.ExplicitExpression(
      penalty_expression=bound_basic_expression,
      constraint_expression=bound_basic_expression)
  extra_constraints = [ee <= bound_expression for ee in expressions]

  # We wrap the result in a BoundedExpression so that we'll check if the user
  # attempts to maximize of lower-bound the result of this function, and will
  # raise an error if they do.
  return expression.BoundedExpression(
      lower_bound=expression.InvalidExpression(
          "the result of a call to upper_bound() can only be minimized or "
          "upper-bounded; it *cannot* be maximized or lower-bounded"),
      upper_bound=expression.ConstrainedExpression(
          expression.ExplicitExpression(
              penalty_expression=bound_basic_expression,
              constraint_expression=bound_basic_expression),
          extra_constraints=extra_constraints))


def lower_bound(expressions):
  """Creates an `Expression` lower bounding the given expressions.

  This function introduces a slack variable, and adds constraints forcing this
  variable to lower bound all elements of the given expression list. It then
  returns the slack variable.

  If you're going to be lower-bounding or maximizing the result of this
  function, then you can think of it as taking the `min` of its arguments. You
  should *never* upper-bound or minimize the result, however, since the
  consequence would be to decrease the value of the slack variable, without
  affecting the contents of the expressions list.

  Args:
    expressions: list of `Expression`s, the quantities to lower-bound.

  Returns:
    An `Expression` representing an lower bound on the given expressions.

  Raises:
    ValueError: if the expressions list is empty.
    TypeError: if the expressions list contains a non-`Expression`.
  """
  if not expressions:
    raise ValueError("lower_bound cannot be given an empty expression list")
  if not all(isinstance(ee, expression.Expression) for ee in expressions):
    raise TypeError(
        "lower_bound expects a list of rate Expressions (perhaps you need to "
        "call wrap_rate() to create an Expression from a Tensor?)")

  # Ideally the slack variable would have the same dtype as the predictions, but
  # we might not know their dtype (e.g. in eager mode), so instead we always use
  # float32 with auto_cast=True.
  bound = deferred_tensor.DeferredVariable(
      0.0,
      trainable=True,
      name="tfco_lower_bound",
      dtype=tf.float32,
      auto_cast=True)

  bound_basic_expression = basic_expression.BasicExpression(
      [term.TensorTerm(bound)])
  bound_expression = expression.ExplicitExpression(
      penalty_expression=bound_basic_expression,
      constraint_expression=bound_basic_expression)
  extra_constraints = [ee >= bound_expression for ee in expressions]

  # We wrap the result in a BoundedExpression so that we'll check if the user
  # attempts to minimize of upper-bound the result of this function, and will
  # raise an error if they do.
  return expression.BoundedExpression(
      lower_bound=expression.ConstrainedExpression(
          expression.ExplicitExpression(
              penalty_expression=bound_basic_expression,
              constraint_expression=bound_basic_expression),
          extra_constraints=extra_constraints),
      upper_bound=expression.InvalidExpression(
          "the result of a call to lower_bound() can only be maximized or "
          "lower-bounded; it *cannot* be minimized or upper-bounded"))
