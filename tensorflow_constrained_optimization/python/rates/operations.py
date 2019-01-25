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

import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import expression


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
  # Tensors, or are types that can be converted to scalar Tensors.
  # Unfortunately, this includes a lot of possible types, so the easiest
  # solution would be to actually perform the conversion, and then check that
  # the resulting Tensor has only one element. This, however, would add a dummy
  # element to the Tensorflow graph, and wouldn't work for a Tensor with an
  # unknown size. Hence, we check only the most common failure case (trying to
  # wrap an Expression).
  if (isinstance(penalty_tensor, expression.Expression) or
      isinstance(constraint_tensor, expression.Expression)):
    raise TypeError("you cannot wrap an object that has already been wrapped")

  penalty_expression = basic_expression.BasicExpression(
      terms=[], tensor=penalty_tensor)
  if constraint_tensor is None:
    constraint_expression = penalty_expression
  else:
    constraint_expression = basic_expression.BasicExpression(
        terms=[], tensor=constraint_tensor)
  return expression.Expression(penalty_expression, constraint_expression)


def _create_slack_variable(expressions, name):
  """Creates a slack variable for upper_bound() or lower_bound().

  Args:
    expressions: list of `Expression`s, the quantities to lower-bound.
    name: name of the function (either "upper_bound" or "lower_bound") calling
      this function. Will also be used to name the resulting TensorFlow
      variable.

  Returns:
    A TensorFlow variable to use as the slack variable in upper_bound() or
      lower_bound().

  Raises:
    ValueError: if the expressions list is empty.
    TypeError: if the expressions list contains a non-`Expression`, or if any
      `Expression` has a different dtype.
  """
  if not expressions:
    raise ValueError("%s cannot be given an empty expression list" % name)
  if not all(isinstance(ee, expression.Expression) for ee in expressions):
    raise TypeError(
        "%s expects a list of rate Expressions (perhaps you need to call "
        "wrap_rate() to create an Expression from a Tensor?)" % name)

  # The dtype of the slack variable will be the same as the dtypes of the
  # Expressions it's bounding. If the expressions don't have the same dtypes,
  # then we don't know what dtype the slack variable should be, so we raise an
  # error.
  dtypes = set(ee.penalty_expression.dtype for ee in expressions)
  dtypes.update(ee.constraint_expression.dtype for ee in expressions)
  if len(dtypes) != 1:
    raise TypeError(
        "all Expressions passed to %s must have the same dtype" % name)
  dtype = dtypes.pop()
  return tf.Variable(0.0, dtype=dtype, name=name)


def upper_bound(expressions):
  """Creates an `Expression` upper bounding the given expressions.

  This function introduces a slack variable, and adds constraints forcing this
  variable to upper bound all elements of the given expression list. It then
  returns the slack variable.

  If you're going to be upper-bounding or minimizing the result of this
  function, then you can think of it as taking the `max` of its arguments. It's
  different from `max` if you're going to be lower-bounding or maximizing the
  result, however, since the consequence would be to increase the value of the
  slack variable, without affecting the contents of the expressions list.

  Args:
    expressions: list of `Expression`s, the quantities to upper-bound.

  Returns:
    An `Expression` representing an upper bound on the given expressions.

  Raises:
    ValueError: if the expressions list is empty.
    TypeError: if the expressions list contains a non-`Expression`, or if any
      `Expression` has a different dtype.
  """
  bound = _create_slack_variable(expressions, "upper_bound")

  bound_expression = basic_expression.BasicExpression(terms=[], tensor=bound)
  extra_constraints = set(ee <= bound for ee in set(expressions))
  return expression.Expression(
      penalty_expression=bound_expression,
      constraint_expression=bound_expression,
      extra_constraints=extra_constraints)


def lower_bound(expressions):
  """Creates an `Expression` lower bounding the given expressions.

  This function introduces a slack variable, and adds constraints forcing this
  variable to lower bound all elements of the given expression list. It then
  returns the slack variable.

  If you're going to be lower-bounding or maximizing the result of this
  function, then you can think of it as taking the `min` of its arguments. It's
  different from `min` if you're going to be upper-bounding or minimizing the
  result, however, since the consequence would be to decrease the value of the
  slack variable, without affecting the contents of the expressions list.

  Args:
    expressions: list of `Expression`s, the quantities to lower-bound.

  Returns:
    An `Expression` representing an lower bound on the given expressions.

  Raises:
    ValueError: if the expressions list is empty.
    TypeError: if the expressions list contains a non-`Expression`, or if any
      `Expression` has a different dtype.
  """
  bound = _create_slack_variable(expressions, "lower_bound")

  bound_expression = basic_expression.BasicExpression(terms=[], tensor=bound)
  extra_constraints = set(ee >= bound for ee in set(expressions))
  return expression.Expression(
      penalty_expression=bound_expression,
      constraint_expression=bound_expression,
      extra_constraints=extra_constraints)
