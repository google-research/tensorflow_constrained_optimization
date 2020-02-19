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
"""Contains functions for constructing binary classification rate expressions.

A rate constraints problem is constructed as an objective function and set of
constraints, all of which are represented as `Expression`s, each of which
represents a linear combination of `Term`s (which represent rates) and
`Tensor`s.

Rates are things like the error rate, the false positive rate, and so on. The
functions in this file are responsible for constructing the objects that
represent such rates, which then are minimized or constrained to form an
optimization problem.

All rate-constructing functions return `Expression`s, each of which contains
*two* different approximations to the desired rate. The "penalty"
`BasicExpression` is an approximation to using penalty_loss, while the
"constraint" `BasicExpression` is based on constraint_loss (if constraint_loss
is the zero-one loss, which is the default, then the "constraint" expression
will be exactly total_rate as defined above, with no approximation).

The reason an `Expression` contains two `BasicExpression`s is that the
"penalty" `BasicExpression` will be differentiable, while the "constraint"
`BasicExpression` need not be. During optimization, the former will be used
whenever we need to take gradients, and the latter otherwise.

Example
=======

Suppose that "examples_tensor" and "labels_tensor" are placeholder `Tensor`s
containing training examples and their associated labels, and the "model"
function evaluates the model you're trying to train, returning a `Tensor` of
model evaluations. Further suppose that we wish to impose a fairness-style
constraint on a protected class (the "blue" examples). The following code will
create three contexts (see subsettable_context.py) representing all of the
examples, and only the "blue" examples, respectively.

```python
ctx = rate_context(model(examples_tensor), labels_tensor)
blue_ctx = ctx.subset(examples_tensor[:, is_blue_idx])
```

Now that we have the contexts, we can create the rates. We'll try to minimize
the overall error rate, while constraining the true positive rate on the "blue"
class to be between 90% and 110% of the overall true positive rate:

```python
objective = error_rate(ctx)
constraints = [
    true_positive_rate(blue_ctx) >= 0.9 * true_positive_rate(ctx),
    true_positive_rate(blue_ctx) <= 1.1 * true_positive_rate(ctx)
]
```

The error_rate() and true_positive_rate() functions are two of the
rate-constructing functions that are defined in this file. This objective and
list of constraints can then be passed on to the MinimizationProblem constructor
in rate_minimization_problem.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import expression
from tensorflow_constrained_optimization.python.rates import loss
from tensorflow_constrained_optimization.python.rates import subsettable_context
from tensorflow_constrained_optimization.python.rates import term


def _binary_classification_rate(
    positive_coefficient=0.0,
    negative_coefficient=0.0,
    numerator_context=None,
    denominator_context=None,
    penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
    constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing positive and negative rates.

  The result of this function represents:
    total_rate := (positive_coefficient * positive_rate +
        negative_coefficient * negative_rate)
  where:
    positive_rate := sum_i{w_i * c_i * d_i * 1{z_i >  0}} / sum_i{w_i * d_i}
    negative_rate := sum_i{w_i * c_i * d_i * 1{z_i <= 0}} / sum_i{w_i * d_i}
  where z_i and w_i are the given predictions and weights, and c_i and d_i are
  indicators for which examples to include the numerator and denominator (all
  four of z, w, c and d are in the contexts).

  Args:
    positive_coefficient: float, scalar coefficient on the positive prediction
      rate.
    negative_coefficient: float, scalar coefficient on the negative prediction
      rate.
    numerator_context: `SubsettableContext`, the block of data to use when
      calculating the numerators of the rates.
    denominator_context: `SubsettableContext`, the block of data to use when
      calculating the denominators of the rates.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rates.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rates.

  Returns:
    An `Expression` representing total_rate (as defined above).

  Raises:
    TypeError: if either context is not a SubsettableContext, or either loss is
      not a BinaryClassificationLoss.
    ValueError: if either context is not provided, or the two contexts are
      incompatible (have different predictions, labels or weights).
  """
  if numerator_context is None or denominator_context is None:
    raise ValueError("both numerator_context and denominator_context must be "
                     "provided")
  if not (
      isinstance(numerator_context, subsettable_context.SubsettableContext) and
      isinstance(denominator_context, subsettable_context.SubsettableContext)):
    raise TypeError("numerator and denominator contexts must be "
                    "SubsettableContext objects")
  raw_context = numerator_context.raw_context
  if denominator_context.raw_context != raw_context:
    raise ValueError("numerator and denominator contexts must be compatible")

  if not (isinstance(penalty_loss, loss.BinaryClassificationLoss) and
          isinstance(constraint_loss, loss.BinaryClassificationLoss)):
    raise TypeError("penalty and constraint losses must be "
                    "BinaryClassificationLoss objects")

  penalty_term = term.BinaryClassificationTerm.ratio(
      positive_coefficient, negative_coefficient,
      raw_context.penalty_predictions, raw_context.penalty_weights,
      numerator_context.penalty_predicate,
      denominator_context.penalty_predicate, penalty_loss)
  constraint_term = term.BinaryClassificationTerm.ratio(
      positive_coefficient, negative_coefficient,
      raw_context.constraint_predictions, raw_context.constraint_weights,
      numerator_context.constraint_predicate,
      denominator_context.constraint_predicate, constraint_loss)

  return expression.ExplicitExpression(
      basic_expression.BasicExpression([penalty_term]),
      basic_expression.BasicExpression([constraint_term]))


def _ratio_bound(numerator_expression, denominator_expression, lower_bound,
                 upper_bound):
  """Creates an `Expression` representing a ratio.

  The result of this function is an `Expression` representing:
    numerator_bound / denominator_bound
  where numerator_bound and denominator_bound are two newly-created slack
  variables projected to satisfy the following in an update op:
    0 <= numerator_bound <= denominator_bound <= 1
    denominator_lower_bound <= denominator_bound
  Additionally, the following two constraints will be added if lower_bound is
  True:
    numerator_bound <= numerator_expression
    denominator_bound >= denominator_expression
  and/or the following two if upper_bound is true:
    numerator_bound >= numerator_expression
    denominator_bound <= denominator_expression
  These constraints are placed in the "extra_constraints" field of the resulting
  `Expression`.

  If you're going to be lower-bounding or maximizing the result of this
  function, then need to set the lower_bound parameter to `True`. Likewise, if
  you're going to be upper-bounding or minimizing the result of this function,
  then the upper_bound parameter must be `True`. At least one of these
  parameters *must* be `True`, and it's permitted for both of them to be `True`
  (but we recommend against this, since it would result in equality constraints,
  which might cause problems during optimization and/or post-processing).

  Args:
    numerator_expression: `Expression`, the numerator of the ratio.
    denominator_expression: `Expression`, the denominator of the ratio.
    lower_bound: bool, `True` if you want the result of this function to
      lower-bound the ratio.
    upper_bound: bool, `True` if you want the result of this function to
      upper-bound the ratio.

  Returns:
    An `Expression` representing the ratio.

  Raises:
    TypeError: if either numerator_expression or denominator_expression is not
      an `Expression`.
    ValueError: if both lower_bound and upper_bound are `False`.
  """
  if not (isinstance(numerator_expression, expression.Expression) and
          isinstance(denominator_expression, expression.Expression)):
    raise TypeError(
        "both numerator_expression and denominator_expression must be "
        "Expressions (perhaps you need to call wrap_rate() to create an "
        "Expression from a Tensor?)")

  # One could set both lower_bound and upper_bound to True, in which case the
  # result of this function could be treated as the ratio itself (instead of a
  # {lower,upper} bound of it). However, this would come with some drawbacks: it
  # would of course make optimization more difficult, but more importantly, it
  # would potentially cause post-processing for feasibility (e.g. using
  # "shrinking") to fail to find a feasible solution.
  if not (lower_bound or upper_bound):
    raise ValueError("at least one of lower_bound or upper_bound must be True")

  # We use a "update_ops_fn" instead of a "constraint" (which we would usually
  # prefer) to perform the projection because we want to grab the denominator
  # lower bound out of the memoizer.
  def update_ops_fn(ratio_bounds_variable, memoizer):
    """Projects ratio_bounds onto the feasible region."""
    numerator = ratio_bounds_variable[0]
    denominator = ratio_bounds_variable[1]

    # First make sure that numerator <= denominator.
    average = 0.5 * (numerator + denominator)
    numerator = tf.minimum(average, numerator)
    denominator = tf.maximum(average, denominator)

    # Next make sure that the numerator is in [0, 1] and the denominator is in
    # [denominator_lower_bound, 1].
    numerator = tf.maximum(0.0, tf.minimum(1.0, numerator))
    denominator = tf.maximum(memoizer[defaults.DENOMINATOR_LOWER_BOUND_KEY],
                             tf.minimum(1.0, denominator))

    return [ratio_bounds_variable.assign([numerator, denominator])]

  # Ideally the slack variables would have the same dtype as the predictions,
  # but we might not know their dtype (e.g. in eager mode), so instead we always
  # use float32 with auto_cast=True.
  ratio_bounds = deferred_tensor.DeferredVariable([0.0, 1.0],
                                                  trainable=True,
                                                  name="tfco_ratio_bounds",
                                                  dtype=tf.float32,
                                                  update_ops_fn=update_ops_fn,
                                                  auto_cast=True)
  numerator_bound_basic_expression = basic_expression.BasicExpression(
      [term.TensorTerm(ratio_bounds[0])])
  numerator_bound_expression = expression.ExplicitExpression(
      penalty_expression=numerator_bound_basic_expression,
      constraint_expression=numerator_bound_basic_expression)

  denominator_bound_basic_expression = basic_expression.BasicExpression(
      [term.TensorTerm(ratio_bounds[1])])
  denominator_bound_expression = expression.ExplicitExpression(
      penalty_expression=denominator_bound_basic_expression,
      constraint_expression=denominator_bound_basic_expression)

  extra_constraints = []
  if lower_bound:
    extra_constraints.append(numerator_bound_expression <= numerator_expression)
    extra_constraints.append(
        denominator_expression <= denominator_bound_expression)
  if upper_bound:
    extra_constraints.append(numerator_expression <= numerator_bound_expression)
    extra_constraints.append(
        denominator_bound_expression <= denominator_expression)

  ratio_basic_expression = basic_expression.BasicExpression(
      [term.TensorTerm(ratio_bounds[0] / ratio_bounds[1])])
  return expression.ConstrainedExpression(
      expression.ExplicitExpression(
          penalty_expression=ratio_basic_expression,
          constraint_expression=ratio_basic_expression),
      extra_constraints=extra_constraints)


def positive_prediction_rate(context,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing a positive prediction rate.

  The result of this function represents:
    positive_rate := sum_i{w_i * c_i * 1{z_i > 0}} / sum_i{w_i * c_i}
  where z_i and w_i are the given predictions and weights, and c_i is an
  indicator for which examples to include in the rate (all three of z, w and c
  are in the context).

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing positive_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
  """
  return _binary_classification_rate(
      positive_coefficient=1.0,
      numerator_context=context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def negative_prediction_rate(context,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing a negative prediction rate.

  The result of this function represents:
    negative_rate := sum_i{w_i * c_i * 1{z_i <= 0}} / sum_i{w_i * c_i}
  where z_i and w_i are the given predictions and weights, and c_i is an
  indicator for which examples to include in the rate (all three of z, w and c
  are in the context).

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing negative_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
  """
  return _binary_classification_rate(
      negative_coefficient=1.0,
      numerator_context=context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def error_rate(context,
               penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
               constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing an error rate.

  The result of this function represents:

  $$\\mathrm{error\_rate} = \\frac{
    \sum_i w_i c_i ( \\mathbf{1}\{y_i \\le 0 \\wedge z_i > 0\} +
    \\mathbf{1}\{y_i > 0 \\wedge z_i \\le 0\} )
  }{ \sum_i w_i c_i }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing error_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("error_rate requires a context with labels")

  positive_context = context.subset(raw_context.penalty_labels > 0,
                                    raw_context.constraint_labels > 0)
  negative_context = context.subset(raw_context.penalty_labels <= 0,
                                    raw_context.constraint_labels <= 0)

  positive_expression = _binary_classification_rate(
      negative_coefficient=1.0,
      numerator_context=positive_context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)
  negative_expression = _binary_classification_rate(
      positive_coefficient=1.0,
      numerator_context=negative_context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)

  return positive_expression + negative_expression


def accuracy_rate(context,
                  penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                  constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing an accuracy rate.

  The result of this function represents:

  $$\\mathrm{accuracy\_rate} = \\frac{
    \sum_i w_i c_i ( \\mathbf{1}\{y_i > 0 \\wedge z_i > 0\} +
    \\mathbf{1}\{y_i \\le 0 \\wedge z_i \\le 0\} )
  }{ \sum_i w_i c_i }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing accuracy_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("accuracy_rate requires a context with labels")

  positive_context = context.subset(raw_context.penalty_labels > 0,
                                    raw_context.constraint_labels > 0)
  negative_context = context.subset(raw_context.penalty_labels <= 0,
                                    raw_context.constraint_labels <= 0)

  positive_expression = _binary_classification_rate(
      positive_coefficient=1.0,
      numerator_context=positive_context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)
  negative_expression = _binary_classification_rate(
      negative_coefficient=1.0,
      numerator_context=negative_context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)

  return positive_expression + negative_expression


def true_positive_rate(context,
                       penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                       constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing a true positive rate.

  The result of this function represents:

  $$\\mathrm{true\_positive\_rate} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{y_i > 0\} }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Unlike the "true_positive_proportion", which is the number of
  positively-labeled examples on which we make a positive prediction, divided by
  the total number of examples, this function instead divides only by the number
  of *positively-labeled* examples.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing true_positive_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("true_positive_rate requires a context with labels")

  positive_context = context.subset(raw_context.penalty_labels > 0,
                                    raw_context.constraint_labels > 0)

  return _binary_classification_rate(
      positive_coefficient=1.0,
      numerator_context=positive_context,
      denominator_context=positive_context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_negative_rate(context,
                        penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                        constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing a false negative rate.

  The result of this function represents:

  $$\\mathrm{false\_negative\_rate} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i \le 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{y_i > 0\} }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Unlike the "false_negative_proportion", which is the number of
  positively-labeled examples on which we make a negative prediction, divided by
  the total number of examples, this function instead divides only by the number
  of *positively-labeled* examples.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing false_negative_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("false_negative_rate requires a context with labels")

  positive_context = context.subset(raw_context.penalty_labels > 0,
                                    raw_context.constraint_labels > 0)

  return _binary_classification_rate(
      negative_coefficient=1.0,
      numerator_context=positive_context,
      denominator_context=positive_context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_positive_rate(context,
                        penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                        constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing a false positive rate.

  The result of this function represents:

  $$\\mathrm{false\_positive\_rate} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i \le 0 \\wedge z_i > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{y_i \le 0\} }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Unlike the "false_positive_proportion", which is the number of
  negatively-labeled examples on which we make a positive prediction, divided by
  the total number of examples, this function instead divides only by the number
  of *negatively-labeled* examples.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing false_positive_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("false_positive_rate requires a context with labels")

  negative_context = context.subset(raw_context.penalty_labels <= 0,
                                    raw_context.constraint_labels <= 0)

  return _binary_classification_rate(
      positive_coefficient=1.0,
      numerator_context=negative_context,
      denominator_context=negative_context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def true_negative_rate(context,
                       penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                       constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing a true negative rate.

  The result of this function represents:

  $$\\mathrm{true\_negative\_rate} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i \le 0 \\wedge z_i \le 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{y_i \le 0\} }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Unlike the "true_negative_proportion", which is the number of
  negatively-labeled examples on which we make a negative prediction, divided by
  the total number of examples, this function instead divides only by the number
  of *negatively-labeled* examples.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing true_negative_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("true_negative_rate requires a context with labels")

  negative_context = context.subset(raw_context.penalty_labels <= 0,
                                    raw_context.constraint_labels <= 0)

  return _binary_classification_rate(
      negative_coefficient=1.0,
      numerator_context=negative_context,
      denominator_context=negative_context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def true_positive_proportion(context,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing a true positive proportion.

  The result of this function represents:

  $$\\mathrm{true\_positive\_proportion} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i > 0\}
  }{ \sum_i w_i c_i }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Unlike the "true_positive_rate", which is the number of positively-labeled
  examples on which we make a positive prediction, divided by the number of
  positively-labeled examples, this function instead divides by the total number
  of examples.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing true_positive_proportion (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("true_positive_proportion requires a context with labels")

  positive_context = context.subset(raw_context.penalty_labels > 0,
                                    raw_context.constraint_labels > 0)

  return _binary_classification_rate(
      positive_coefficient=1.0,
      numerator_context=positive_context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_negative_proportion(context,
                              penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                              constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing a false negative proportion.

  The result of this function represents:

  $$\\mathrm{false\_negative\_proportion} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i \le 0\}
  }{ \sum_i w_i c_i }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Unlike the "false_negative_rate", which is the number of positively-labeled
  examples on which we make a negative prediction, divided by the number of
  positively-labeled examples, this function instead divides by the total number
  of examples.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing false_negative_proportion (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("false_negative_proportion requires a context with labels")

  positive_context = context.subset(raw_context.penalty_labels > 0,
                                    raw_context.constraint_labels > 0)

  return _binary_classification_rate(
      negative_coefficient=1.0,
      numerator_context=positive_context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_positive_proportion(context,
                              penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                              constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing a false positive proportion.

  The result of this function represents:

  $$\\mathrm{false\_positive\_proportion} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i \le 0 \\wedge z_i > 0\}
  }{ \sum_i w_i c_i }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Unlike the "false_positive_rate", which is the number of negatively-labeled
  examples on which we make a positive prediction, divided by the number of
  negatively-labeled examples, this function instead divides by the total number
  of examples.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing false_positive_proportion (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("false_positive_proportion requires a context with labels")

  negative_context = context.subset(raw_context.penalty_labels <= 0,
                                    raw_context.constraint_labels <= 0)

  return _binary_classification_rate(
      positive_coefficient=1.0,
      numerator_context=negative_context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def true_negative_proportion(context,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing a true negative proportion.

  The result of this function represents:

  $$\\mathrm{true\_negative\_proportion} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i \le 0 \\wedge z_i \le 0\}
  }{ \sum_i w_i c_i }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Unlike the "true_negative_rate", which is the number of negatively-labeled
  examples on which we make a negative prediction, divided by the number of
  negatively-labeled examples, this function instead divides by the total number
  of examples.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing true_negative_proportion (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("true_negative_proportion requires a context with labels")

  negative_context = context.subset(raw_context.penalty_labels <= 0,
                                    raw_context.constraint_labels <= 0)

  return _binary_classification_rate(
      negative_coefficient=1.0,
      numerator_context=negative_context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def precision_ratio(context,
                    penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                    constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates two `Expression`s representing a precision as a ratio.

  The result of this function represents:

  $$\\mathrm{numerator} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i > 0\}
  }{ \sum_i w_i c_i }$$

  $$\\mathrm{denominator} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{z_i > 0\}
  }{ \sum_i w_i c_i }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  The reason for decomposing a precision as a separate numerator and denominator
  is to make it easy to set up constraints of the form (for example):

  > precision := numerator / denominator >= 0.9

  for which you can multiply through by the denominator to yield the equivalent
  constraint:

  > numerator >= 0.9 * denominator

  This latter form is something that we can straightforwardly handle.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An (`Expression`, `Expression`) pair representing the numerator and
      denominator of a precision (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("precision requires a context with labels")

  positive_context = context.subset(raw_context.penalty_labels > 0,
                                    raw_context.constraint_labels > 0)

  numerator_expression = _binary_classification_rate(
      positive_coefficient=1.0,
      numerator_context=positive_context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)
  denominator_expression = _binary_classification_rate(
      positive_coefficient=1.0,
      numerator_context=context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)

  return numerator_expression, denominator_expression


def precision(context,
              penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
              constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing precision.

  The result of this function represents:

  $$\\mathrm{precision} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{z_i > 0\} }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing precision (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  numerator_expression, denominator_expression = precision_ratio(
      context, penalty_loss=penalty_loss, constraint_loss=constraint_loss)
  return expression.BoundedExpression(
      lower_bound=_ratio_bound(
          numerator_expression=numerator_expression,
          denominator_expression=denominator_expression,
          lower_bound=True,
          upper_bound=False),
      upper_bound=_ratio_bound(
          numerator_expression=numerator_expression,
          denominator_expression=denominator_expression,
          lower_bound=False,
          upper_bound=True))


def f_score_ratio(context,
                  beta=1.0,
                  penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                  constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates two `Expression`s representing an F-score as a ratio.

  The result of this function represents:

  $$\\mathrm{numerator} = \\frac{
    \sum_i w_i c_i (1 + \\beta^2) \\mathbf{1}\{y_i > 0 \\wedge z_i > 0\}
  }{ \sum_i w_i c_i }$$

  $$\\mathrm{denominator} = \\frac{
    \sum_i w_i c_i ( (1 + \\beta^2) \\mathbf{1}\{y_i > 0 \\wedge z_i > 0\}
    + \\beta^2 \\mathbf{1}\{y_i > 0 \\wedge z_i \\le 0\}
    + \\mathbf{1}\{y_i \\le 0 \\wedge z_i > 0\} )
  }{ \sum_i w_i c_i }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  The reason for decomposing an F-score as a separate numerator and denominator
  is to make it easy to set up constraints of the form (for example):

  > f_score := numerator / denominator >= 0.9

  for which you can multiply through by the denominator to yield the equivalent
  constraint:

  > numerator >= 0.9 * denominator

  This latter form is something that we can straightforwardly handle.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    beta: non-negative float, the beta parameter to the f-score. If beta=0, then
      the result is precision, and if beta=1 (the default), then the result is
      the F1-score.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An (`Expression`, `Expression`) pair representing the numerator and
      denominator of an F-score (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("f_score requires a context with labels")

  if beta < 0.0:
    raise ValueError("beta parameter to f_score must be non-negative")
  if beta <= 0.0:
    # The F0-score is just the precision, and representing it as such results
    # in a slightly simpler expression.
    return precision_ratio(
        context, penalty_loss=penalty_loss, constraint_loss=constraint_loss)

  positive_context = context.subset(raw_context.penalty_labels > 0,
                                    raw_context.constraint_labels > 0)
  negative_context = context.subset(raw_context.penalty_labels <= 0,
                                    raw_context.constraint_labels <= 0)

  numerator_expression = _binary_classification_rate(
      positive_coefficient=(1.0 + beta * beta),
      numerator_context=positive_context,
      denominator_context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)
  denominator_expression = (
      numerator_expression + _binary_classification_rate(
          negative_coefficient=beta * beta,
          numerator_context=positive_context,
          denominator_context=context,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss) + _binary_classification_rate(
              positive_coefficient=1.0,
              numerator_context=negative_context,
              denominator_context=context,
              penalty_loss=penalty_loss,
              constraint_loss=constraint_loss))

  return numerator_expression, denominator_expression


def f_score(context,
            beta=1.0,
            penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
            constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing an F-score.

  The result of this function represents:

  $$\\mathrm{f\_score} = \\frac{
    \sum_i w_i c_i (1 + \\beta^2) \\mathbf{1}\{y_i > 0 \\wedge z_i > 0\}
  }{
    \sum_i w_i c_i ( (1 + \\beta^2) \\mathbf{1}\{y_i > 0 \\wedge z_i > 0\}
    + \\beta^2 \\mathbf{1}\{y_i > 0 \\wedge z_i \\le 0\}
    + \\mathbf{1}\{y_i \\le 0 \\wedge z_i > 0\} )
  }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context).

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    beta: non-negative float, the beta parameter to the f-score. If beta=0, then
      the result is precision, and if beta=1 (the default), then the result is
      the F1-score.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing f_score (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels.
  """
  numerator_expression, denominator_expression = f_score_ratio(
      context, beta, penalty_loss=penalty_loss, constraint_loss=constraint_loss)
  return expression.BoundedExpression(
      lower_bound=_ratio_bound(
          numerator_expression=numerator_expression,
          denominator_expression=denominator_expression,
          lower_bound=True,
          upper_bound=False),
      upper_bound=_ratio_bound(
          numerator_expression=numerator_expression,
          denominator_expression=denominator_expression,
          lower_bound=False,
          upper_bound=True))


def _tpr_at_fpr_bound(context, fpr_target, threshold_tensor, lower_bound,
                      upper_bound, penalty_loss, constraint_loss):
  """Creates an `Expression` representing TPR@FPR.

  This is a helper function for _roc_auc_bound(). It returns an `Expression`
  representing the true positive rate (TPR) at an implicitly-defined threshold
  which is chosen in such a way that the false positive rate (FPR) is either <=
  (if lower_bound is True) or >= (if upper_bound is true) the fpr_target
  parameter.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    fpr_target: float in (0,1), the desired FPR target at which we will fix the
      threshold.
    threshold_tensor: `DeferredTensor`, the parameter to use for the threshold.
    lower_bound: bool, `True` if you want the result of this function to
      lower-bound the TPR@FPR.
    upper_bound: bool, `True` if you want the result of this function to
      upper-bound the TPR@FPR.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate. This loss must be "normalized" (see
      `BinaryClassificationLoss.is_normalized`).

  Returns:
    An `Expression` representing TPR@FPR.

  Raises:
    TypeError: if the context is not a SubsettableContext or either loss is not
      a BinaryClassificationLoss.
    ValueError: if fpr_target isn't in (0,1), both lower_bound and upper_bound
      are `False`, the context doesn't contain labels, or the constraint_loss is
      not normalized.
  """
  if fpr_target <= 0 or fpr_target >= 1:
    raise ValueError("fpr_target must be in (0,1)")

  # One could set both lower_bound and upper_bound to True, in which case the
  # result of this function could be treated as the TPR@FPR itself (instead of a
  # {lower,upper} bound of it). However, this would come with some drawbacks: it
  # would of course make optimization more difficult, but more importantly, it
  # would potentially cause post-processing for feasibility (e.g. using
  # "shrinking") to fail to find a feasible solution.
  if not (lower_bound or upper_bound):
    raise ValueError("at least one of lower_bound or upper_bound must be True")

  # For the constraint on the false positive rates to make sense, it would be
  # best to be using a normalized loss. The reason for this is that, if both
  # lower_bound and upper_bound are True (or if one imposes constraints
  # including separate lower and upper bounds), our constraints on the false
  # positive rates will be equality constraints, which could be infeasible for
  # an unnormalized loss. This could be changed to a warning, however.
  if not constraint_loss.is_normalized:
    raise ValueError("tpr_at_fpr can only be used with a normalized "
                     "constraint_loss (e.g. zero/one, sigmoid or ramp)")

  context = context._transform_predictions(  # pylint: disable=protected-access
      lambda predictions: predictions - threshold_tensor)

  fpr_expression = false_positive_rate(
      context, penalty_loss=penalty_loss, constraint_loss=constraint_loss)

  extra_constraints = []
  if lower_bound:
    extra_constraints.append(fpr_expression <= fpr_target)
  if upper_bound:
    extra_constraints.append(fpr_expression >= fpr_target)

  return expression.ConstrainedExpression(
      true_positive_rate(
          context, penalty_loss=penalty_loss, constraint_loss=constraint_loss),
      extra_constraints=extra_constraints)


def _roc_auc_bound(context, bins, include_threshold, lower_bound, upper_bound,
                   penalty_loss, constraint_loss):
  """Creates an `Expression` representing an approximate ROC AUC.

  The result of this function represents a Riemann approximation to the area
  under the ROC curve (false positive rate on the horizontal axis, true positive
  rate on the vertical axis), using the constraint-based method proposed by:

  > Eban, Schain, Mackey, Gordon, Rifkin and Elidan. "Scalable Learning of
  > Non-Decomposable Objectives". AISTATS 2017.

  If you're going to be lower-bounding or maximizing the result of this
  function, then need to set the lower_bound parameter to `True`. Likewise, if
  you're going to be upper-bounding or minimizing the result of this function
  (which normally wouldn't make much sense for ROC AUC), then the upper_bound
  parameter must be `True`. At least one of these parameters *must* be `True`,
  and it's permitted for both of them to be `True` (but we recommend against
  this, since it would result in equality constraints, which might cause
  problems during optimization and/or post-processing).

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    bins: positive integer, the number of "rectangles" to use for the Riemann
      approximation to ROC AUC.
    include_threshold: if False, the thresholds associated with every bin in the
      Riemann approximation will be constrained to sum to zero. In other words,
      we'll remove one degree of freedom, causing there to be effectively bins-1
      thresholds, instead of bins thresholds.
    lower_bound: bool, `True` if you want the result of this function to
      lower-bound the approximate ROC AUC.
    upper_bound: bool, `True` if you want the result of this function to
      upper-bound the approximate ROC AUC.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate. This loss must be "normalized" (see
      `BinaryClassificationLoss.is_normalized`).

  Returns:
    An `Expression` representing a Riemann approximation to ROC AUC.

  Raises:
    TypeError: if the context is not a SubsettableContext, the number of bins is
      not an integer, or either loss is not a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels, the number of bins is
      non-positive, both lower_bound and upper_bound are `False`, or the
      constraint_loss is not normalized.
  """
  if not isinstance(bins, numbers.Integral):
    raise TypeError("number of roc_auc bins must be an integer")
  if bins <= 0:
    raise ValueError("number of roc_auc bins must be strictly positive")

  constraint = None
  if not include_threshold:
    # If include_threshold is False, then we center the thresholds around zero,
    # effectively causing there to be no overall threshold.
    constraint = lambda tensor: tensor - tf.reduce_mean(tensor)

  # Ideally the thresholds would have the same dtype as the predictions, but we
  # might not know their dtype (e.g. in eager mode), so instead we always use
  # float32 with auto_cast=True.
  thresholds = deferred_tensor.DeferredVariable(
      np.zeros((bins,)),
      trainable=True,
      name="tfco_roc_auc_thresholds",
      dtype=tf.float32,
      constraint=constraint,
      auto_cast=True)

  average_tpr_expression = None
  for bin_index in xrange(bins):
    fpr_target = (bin_index + 0.5) / bins
    tpr_expression = _tpr_at_fpr_bound(
        context,
        fpr_target=fpr_target,
        threshold_tensor=thresholds[bin_index],
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)
    if average_tpr_expression is None:
      average_tpr_expression = tpr_expression
    else:
      average_tpr_expression += tpr_expression

  return average_tpr_expression / bins


def roc_auc(context,
            bins,
            include_threshold=True,
            penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
            constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing an approximate ROC AUC.

  The result of this function represents a Riemann approximation to the area
  under the ROC curve (false positive rate on the horizontal axis, true positive
  rate on the vertical axis), using the constraint-based method proposed by:

  > Eban, Schain, Mackey, Gordon, Rifkin and Elidan. "Scalable Learning of
  > Non-Decomposable Objectives". AISTATS 2017.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    bins: positive integer, the number of "rectangles" to use for the Riemann
      approximation to ROC AUC.
    include_threshold: if False, the thresholds associated with every bin in the
      Riemann approximation will be constrained to sum to zero. In other words,
      we'll remove one degree of freedom, causing there to be effectively bins-1
      thresholds, instead of bins thresholds.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate. This loss must be "normalized" (see
      `BinaryClassificationLoss.is_normalized`).

  Returns:
    An `Expression` representing a Riemann approximation to ROC AUC.

  Raises:
    TypeError: if the context is not a SubsettableContext, the number of bins is
      not an integer, or either loss is not a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels, the number of bins is
      non-positive, or the constraint_loss is not normalized.
  """
  return expression.BoundedExpression(
      lower_bound=_roc_auc_bound(
          context,
          bins,
          include_threshold=include_threshold,
          lower_bound=True,
          upper_bound=False,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss),
      upper_bound=_roc_auc_bound(
          context,
          bins,
          include_threshold=include_threshold,
          lower_bound=False,
          upper_bound=True,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss))


def _recall_at_precision_bound(context, precision_target, include_threshold,
                               lower_bound, upper_bound, penalty_loss,
                               constraint_loss):
  r"""Creates an `Expression` representing recall@precision.

  You should think of the result of this function as "the recall of a
  thresholded classifier, with the threshold being chosen so as to meet a
  precision constraint". In other words, the result is:

  $$\\mathrm{recall(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{y_i > 0\} }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context). The threshold
  $$t$$ is defined in such a way that the precision:

  $$\\mathrm{precision(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{z_i - t > 0\} }$$

  is bounded by the precision_target argument (with the direction of this bound
  being determined by the lower_bound and upper_bound arguments).

  If you're going to be lower-bounding or maximizing the result of this
  function, then need to set the lower_bound parameter to `True`. Likewise, if
  you're going to be upper-bounding or minimizing the result of this function,
  then the upper_bound parameter must be `True`. At least one of these
  parameters *must* be `True`, and it's permitted for both of them to be `True`
  (but we recommend against this, since it would result in equality constraints,
  which might cause problems during optimization and/or post-processing).

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    precision_target: float, the target precision value that will be used to
      define the implicit threshold.
    include_threshold: if False, we will not introduce an implicit threshold at
      which we will constrain the precision and evaluate the recall. Instead, we
      will do so at threshold zero.
    lower_bound: bool, `True` if you want the result of this function to
      lower-bound recall@precision.
    upper_bound: bool, `True` if you want the result of this function to
      upper-bound recall@precision.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing recall@precision (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext or either loss is not
      a BinaryClassificationLoss.
    ValueError: if precision_target isn't in (0,1), both lower_bound and
      upper_bound are `False`, the context doesn't contain labels, or the
      constraint_loss is not normalized.
  """
  if precision_target <= 0 or precision_target >= 1:
    raise ValueError("precision_target must be in (0,1)")

  # One could set both lower_bound and upper_bound to True, in which case the
  # result of this function could be treated as the recall@precision itself
  # (instead of a {lower,upper} bound of it). However, this would come with some
  # drawbacks: it would of course make optimization more difficult, but more
  # importantly, it would potentially cause post-processing for feasibility
  # (e.g. using "shrinking") to fail to find a feasible solution.
  if not (lower_bound or upper_bound):
    raise ValueError("at least one of lower_bound or upper_bound must be True")

  # For the constraints on the precision to make sense, it would be best to be
  # using a normalized loss. The reason for this is that, if both lower_bound
  # and upper_bound are True (or if one imposes constraints including separate
  # lower and upper bounds), our constraints on the precisions will be equality
  # constraints, which could be infeasible for an unnormalized loss. This could
  # be changed to a warning, however.
  if not constraint_loss.is_normalized:
    raise ValueError("recall_at_precision can only be used with a normalized "
                     "constraint_loss (e.g. zero/one, sigmoid or ramp)")

  if include_threshold:
    # Ideally the threshold would have the same dtype as the predictions, but we
    # might not know their dtype (e.g. in eager mode), so instead we always use
    # float32 with auto_cast=True.
    threshold = deferred_tensor.DeferredVariable(
        0.0,
        trainable=True,
        name="tfco_recall_at_precision_threshold",
        dtype=tf.float32,
        auto_cast=True)
    context = context._transform_predictions(  # pylint: disable=protected-access
        lambda predictions: predictions - threshold)

  (precision_numerator_expression,
   precision_denominator_expression) = precision_ratio(
       context, penalty_loss=penalty_loss, constraint_loss=constraint_loss)

  extra_constraints = []
  if lower_bound:
    extra_constraints.append(
        precision_numerator_expression >= precision_target *
        precision_denominator_expression)
  if upper_bound:
    extra_constraints.append(
        precision_numerator_expression <= precision_target *
        precision_denominator_expression)

  # True positive rate = true positives / labeled positives = recall.
  return expression.ConstrainedExpression(
      true_positive_rate(
          context, penalty_loss=penalty_loss, constraint_loss=constraint_loss),
      extra_constraints=extra_constraints)


def recall_at_precision(context,
                        precision_target,
                        include_threshold=True,
                        penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                        constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing recall@precision.

  You should think of the result of this function as "the recall of a
  thresholded classifier, with the threshold being chosen so as to meet a
  precision constraint". In other words, the result is:

  $$\\mathrm{recall(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{y_i > 0\} }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context). The threshold
  $$t$$ is defined in such a way that the precision:

  $$\\mathrm{precision(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{z_i - t > 0\} }$$

  equals the precision_target argument.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    precision_target: float, the target precision value that will be used to
      define the implicit threshold.
    include_threshold: if False, we will not introduce an implicit threshold at
      which we will constrain the precision and evaluate the recall. Instead, we
      will do so at threshold zero.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing recall@precision (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext or either loss is not
      a BinaryClassificationLoss.
    ValueError: if precision_target isn't in (0,1), the context doesn't contain
      labels, or the constraint_loss is not normalized.
  """
  return expression.BoundedExpression(
      lower_bound=_recall_at_precision_bound(
          context,
          precision_target=precision_target,
          include_threshold=include_threshold,
          lower_bound=True,
          upper_bound=False,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss),
      upper_bound=_recall_at_precision_bound(
          context,
          precision_target=precision_target,
          include_threshold=include_threshold,
          lower_bound=False,
          upper_bound=True,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss))


def _inverse_precision_at_recall_bound(context, recall_target,
                                       include_threshold, lower_bound,
                                       upper_bound, penalty_loss,
                                       constraint_loss):
  r"""Creates an `Expression` representing (1/precision)@recall.

  You should think of the result of this function as "the inverse-precision of a
  thresholded classifier, with the threshold being chosen so as to meet a
  recall constraint". In other words, the result is:

  $$\\mathrm{precision^{-1}(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{z_i - t > 0\}
  }{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context). The threshold
  $$t$$ is defined in such a way that the recall:

  $$\\mathrm{recall(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{y_i > 0\} }$$

  is bounded by the recall_target argument (with the direction of this bound
  being determined by the lower_bound and upper_bound arguments).

  If you're going to be lower-bounding or maximizing the result of this
  function, then need to set the lower_bound parameter to `True`. Likewise, if
  you're going to be upper-bounding or minimizing the result of this function,
  then the upper_bound parameter must be `True`. At least one of these
  parameters *must* be `True`, and it's permitted for both of them to be `True`
  (but we recommend against this, since it would result in equality constraints,
  which might cause problems during optimization and/or post-processing).

  The reason for providing this function, in addition to precision_at_recall
  (which could often be used instead), is that this function is "tighter" than
  the other. If you can equivalently reformulate your optimization problem to
  minimize the inverse-precision, instead of maximizing the precision, then you
  should do so.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    recall_target: float, the target recall value that will be used to define
      the implicit threshold.
    include_threshold: if False, we will not introduce an implicit threshold at
      which we will constrain the recall and evaluate the inverse-precision.
      Instead, we will do so at threshold zero.
    lower_bound: bool, `True` if you want the result of this function to
      lower-bound (1/precision)@recall.
    upper_bound: bool, `True` if you want the result of this function to
      upper-bound (1/precision)@recall.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing (1/precision)@recall (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext or either loss is not
      a BinaryClassificationLoss.
    ValueError: if recall_target isn't in (0,1), both lower_bound and
      upper_bound are `False`, the context doesn't contain labels, or the
      constraint_loss is not normalized.
  """

  # Let #TP and #FP be the number of true and false positives, respectively,
  # and take #LP to be the number of examples with positive labels. Then we can
  # write:
  #   recall = #TP / #LP
  #   inverse-precision = (#TP / (#TP + #FP))^-1 = 1 + #FP / #TP
  # Since we're constraining precision at a given recall threshold, we write
  # inverse-precision as:
  #   inverse-precision = 1 + #FP / #TP = 1 + (#FP / #LP) / recall
  # which suggests that we can implement a version of inverse-precision@recall
  # for maximization or lower-bounding as:
  #   1 + (#FP / #LP) / recall_target
  #   s.t. recall_target >= #TP / #LP
  # Likewise, a version for minimization or upper-bounding would be:
  #   1 + (#FP / #LP) / recall_target
  #   s.t. recall_target <= #TP / #LP
  # The direction of these inequalities is chosen due to the fact that, as
  # recall increases, precision will tend to decrease, so inverse_precision
  # will tend to increase (it won't do so monotonically, but it will have that
  # tendency).

  if recall_target <= 0 or recall_target >= 1:
    raise ValueError("recall_target must be in (0,1)")

  # One could set both lower_bound and upper_bound to True, in which case the
  # result of this function could be treated as the inverse-precision@recall
  # itself (instead of a {lower,upper} bound of it). However, this would come
  # with some drawbacks: it would of course make optimization more difficult,
  # but more importantly, it would potentially cause post-processing for
  # feasibility (e.g. using "shrinking") to fail to find a feasible solution.
  if not (lower_bound or upper_bound):
    raise ValueError("at least one of lower_bound or upper_bound must be True")

  # For the constraints on the recalls to make sense, it would be best to be
  # using a normalized loss. The reason for this is that, if both lower_bound
  # and upper_bound are True (or if one imposes constraints including separate
  # lower and upper bounds), our constraints on the recalls will be equality
  # constraints, which could be infeasible for an unnormalized loss. This could
  # be changed to a warning, however.
  if not constraint_loss.is_normalized:
    raise ValueError("inverse_precision_at_recall can only be used with a "
                     "normalized constraint_loss (e.g. zero/one, sigmoid or "
                     "ramp)")

  if include_threshold:
    # Ideally the threshold would have the same dtype as the predictions, but we
    # might not know their dtype (e.g. in eager mode), so instead we always use
    # float32 with auto_cast=True.
    threshold = deferred_tensor.DeferredVariable(
        0.0,
        trainable=True,
        name="tfco_inverse_precision_at_recall_threshold",
        dtype=tf.float32,
        auto_cast=True)
    context = context._transform_predictions(  # pylint: disable=protected-access
        lambda predictions: predictions - threshold)

  if not isinstance(context, subsettable_context.SubsettableContext):
    raise TypeError("context must be a SubsettableContext object")
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("inverse_precision_at_recall requires a context with "
                     "labels")

  positive_context = context.subset(raw_context.penalty_labels > 0,
                                    raw_context.constraint_labels > 0)
  negative_context = context.subset(raw_context.penalty_labels <= 0,
                                    raw_context.constraint_labels <= 0)

  recall_expression = _binary_classification_rate(
      positive_coefficient=1.0,
      negative_coefficient=0.0,
      numerator_context=positive_context,
      denominator_context=positive_context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)

  extra_constraints = []
  if lower_bound:
    extra_constraints.append(recall_expression <= recall_target)
  if upper_bound:
    extra_constraints.append(recall_expression >= recall_target)

  return expression.ConstrainedExpression(
      1 + _binary_classification_rate(
          positive_coefficient=1.0,
          negative_coefficient=0.0,
          numerator_context=negative_context,
          denominator_context=positive_context,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss) / recall_target,
      extra_constraints=extra_constraints)


def inverse_precision_at_recall(
    context,
    recall_target,
    include_threshold=True,
    penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
    constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing (1/precision)@recall.

  You should think of the result of this function as "the inverse-precision of a
  thresholded classifier, with the threshold being chosen so as to meet a recall
  constraint". In other words, the result is:

  $$\\mathrm{precision^{-1}(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{z_i - t > 0\}
  }{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context). The threshold
  $$t$$ is defined in such a way that the recall:

  $$\\mathrm{recall(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{y_i > 0\} }$$

  equals the recall_target argument.

  The reason for providing this function, in addition to precision_at_recall
  (which could often be used instead), is that this function results in a
  simpler optimization than the other. If you can equivalently reformulate your
  optimization problem to maximize the inverse-precision, instead of minimizing
  the precision, then you should do so.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    recall_target: float, the target recall value that will be used to define
      the implicit threshold.
    include_threshold: if False, we will not introduce an implicit threshold at
      which we will constrain the recall and evaluate the inverse-precision.
      Instead, we will do so at threshold zero.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing (1/precision)@recall (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext or either loss is not
      a BinaryClassificationLoss.
    ValueError: if recall_target isn't in (0,1), the context doesn't contain
      labels, or the constraint_loss is not normalized.
  """
  return expression.BoundedExpression(
      lower_bound=_inverse_precision_at_recall_bound(
          context,
          recall_target=recall_target,
          include_threshold=include_threshold,
          lower_bound=True,
          upper_bound=False,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss),
      upper_bound=_inverse_precision_at_recall_bound(
          context,
          recall_target=recall_target,
          include_threshold=include_threshold,
          lower_bound=False,
          upper_bound=True,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss))


def _precision_at_recall_bound(context, recall_target, threshold_tensor,
                               slack_tensor, lower_bound, upper_bound,
                               penalty_loss, constraint_loss):
  r"""Creates an `Expression` representing precision@recall.

  You should think of the result of this function as "the precision of a
  thresholded classifier, with the threshold being chosen so as to meet a
  recall constraint". In other words, the result is:

  $$\\mathrm{precision(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{z_i - t > 0\} }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context). The threshold
  $$t$$ is defined in such a way that the recall:

  $$\\mathrm{recall(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{y_i > 0\} }$$

  is bounded by the recall_target argument (with the direction of this bound
  being determined by the lower_bound and upper_bound arguments).

  If you're going to be lower-bounding or maximizing the result of this
  function, then need to set the lower_bound parameter to `True`. Likewise, if
  you're going to be upper-bounding or minimizing the result of this function,
  then the upper_bound parameter must be `True`. At least one of these
  parameters *must* be `True`, and it's permitted for both of them to be `True`
  (but we recommend against this, since it would result in equality constraints,
  which might cause problems during optimization and/or post-processing).

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    recall_target: float, the target recall value that will be used to define
      the implicit threshold.
    threshold_tensor: `DeferredTensor`, the parameter to use for the threshold.
      If threshold_tensor=None, then the effect is the same as the
      include_threshold=False argument to some of the other functions in this
      file, in that we will not introduce an implicit threshold at which we will
      constrain the recall and evaluate the precision. Instead, we will do so at
      threshold zero.
    slack_tensor: `DeferredTensor`, the parameter to use for the slack variable.
    lower_bound: bool, `True` if you want the result of this function to
      lower-bound precision@recall.
    upper_bound: bool, `True` if you want the result of this function to
      upper-bound precision@recall.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing precision@recall (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext or either loss is not
      a BinaryClassificationLoss.
    ValueError: if recall_target isn't in (0,1), both lower_bound and
      upper_bound are `False`, the context doesn't contain labels, or the
      constraint_loss is not normalized.
  """

  # Let #TP and #FP be the number of true and false positives, respectively,
  # and take #LP to be the number of examples with positive labels. Then we can
  # write:
  #   recall = #TP / #LP
  #   precision = #TP / (#TP + #FP)
  # Since we're constraining precision at a given recall threshold, we write
  # precision as:
  #   precision = (#TP / #LP) / ((#TP / #LP) + (#FP / #LP))
  #             = recall / (recall + (#FP / #LP))
  # which suggests that we can implement a version of precision@recall for
  # maximization or lower-bounding as:
  #   recall_target / (recall_target + slack)
  #   s.t. recall_target >= #TP / #LP
  #        slack >= #FP / #LP
  # Likewise, a version for minimization or upper-bounding would be:
  #   recall_target / (recall_target + slack)
  #   s.t. recall_target <= #TP / #LP
  #        slack <= #FP / #LP
  # The direction of these inequalities is chosen due to the fact that, as
  # recall increases, precision will tend to decrease (it won't do so
  # monotonically, but it will have that tendency).

  if recall_target <= 0 or recall_target >= 1:
    raise ValueError("recall_target must be in (0,1)")

  # One could set both lower_bound and upper_bound to True, in which case the
  # result of this function could be treated as the precision@recall itself
  # (instead of a {lower,upper} bound of it). However, this would come with some
  # drawbacks: it would of course make optimization more difficult, but more
  # importantly, it would potentially cause post-processing for feasibility
  # (e.g. using "shrinking") to fail to find a feasible solution.
  if not (lower_bound or upper_bound):
    raise ValueError("at least one of lower_bound or upper_bound must be True")

  # For the constraints on the recalls to make sense, it would be best to be
  # using a normalized loss. The reason for this is that, if both lower_bound
  # and upper_bound are True (or if one imposes constraints including separate
  # lower and upper bounds), our constraints on the recalls will be equality
  # constraints, which could be infeasible for an unnormalized loss. This could
  # be changed to a warning, however.
  if not constraint_loss.is_normalized:
    raise ValueError("precision_at_recall can only be used with a normalized "
                     "constraint_loss (e.g. zero/one, sigmoid or ramp)")

  if threshold_tensor is not None:
    context = context._transform_predictions(  # pylint: disable=protected-access
        lambda predictions: predictions - threshold_tensor)

  if not isinstance(context, subsettable_context.SubsettableContext):
    raise TypeError("context must be a SubsettableContext object")
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("precision_at_recall requires a context with labels")

  positive_context = context.subset(raw_context.penalty_labels > 0,
                                    raw_context.constraint_labels > 0)
  negative_context = context.subset(raw_context.penalty_labels <= 0,
                                    raw_context.constraint_labels <= 0)

  recall_expression = _binary_classification_rate(
      positive_coefficient=1.0,
      negative_coefficient=0.0,
      numerator_context=positive_context,
      denominator_context=positive_context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)

  constraint_expression = _binary_classification_rate(
      positive_coefficient=1.0,
      negative_coefficient=0.0,
      numerator_context=negative_context,
      denominator_context=positive_context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)

  slack_basic_expression = basic_expression.BasicExpression(
      [term.TensorTerm(slack_tensor)])
  slack_expression = expression.ExplicitExpression(
      penalty_expression=slack_basic_expression,
      constraint_expression=slack_basic_expression)

  extra_constraints = []
  if lower_bound:
    extra_constraints.append(recall_expression >= recall_target)
    extra_constraints.append(slack_expression >= constraint_expression)
  if upper_bound:
    extra_constraints.append(recall_expression <= recall_target)
    extra_constraints.append(slack_expression <= constraint_expression)

  precision_basic_expression = basic_expression.BasicExpression(
      [term.TensorTerm(recall_target / (recall_target + slack_tensor))])
  return expression.ConstrainedExpression(
      expression.ExplicitExpression(
          penalty_expression=precision_basic_expression,
          constraint_expression=precision_basic_expression),
      extra_constraints=extra_constraints)


def precision_at_recall(context,
                        recall_target,
                        include_threshold=True,
                        penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                        constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  r"""Creates an `Expression` representing precision@recall.

  You should think of the result of this function as "the precision of a
  thresholded classifier, with the threshold being chosen so as to meet a recall
  constraint". In other words, the result is:

  $$\\mathrm{precision(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{z_i - t > 0\} }$$

  where $$z_i$$, $$y_i$$ and $$w_i$$ are the given predictions, labels and
  weights, and $$c_i$$ is an indicator for which examples to include in the rate
  (all four of $$z$$, $$y$$, $$w$$ and $$c$$ are in the context). The threshold
  $$t$$ is defined in such a way that the recall:

  $$\\mathrm{recall(t)} = \\frac{
    \sum_i w_i c_i \\mathbf{1}\{y_i > 0 \\wedge z_i - t > 0\}
  }{ \sum_i w_i c_i \\mathbf{1}\{y_i > 0\} }$$

  equals the recall_target argument.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    recall_target: float, the target recall value that will be used to define
      the implicit threshold.
    include_threshold: if False, we will not introduce an implicit threshold at
      which we will constrain the recall and evaluate the precision. Instead, we
      will do so at threshold zero.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate.

  Returns:
    An `Expression` representing precision@recall (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext or either loss is not
      a BinaryClassificationLoss.
    ValueError: if recall_target isn't in (0,1), the context doesn't contain
      labels, or the constraint_loss is not normalized.
  """
  # Ideally the threshold and slack variable would have the same dtype as the
  # predictions, but we might not know their dtype (e.g. in eager mode), so
  # instead we always use float32 with auto_cast=True.
  #
  # We use separate thresholds and slacks for the lower and upper bounds in case
  # the user decides to both lower- and upper-bound the resulting Expression.
  lower_bound_threshold = None
  upper_bound_threshold = None
  if include_threshold:
    lower_bound_threshold = deferred_tensor.DeferredVariable(
        0.0,
        trainable=True,
        name="tfco_precision_at_recall_threshold",
        dtype=tf.float32,
        auto_cast=True)
    upper_bound_threshold = deferred_tensor.DeferredVariable(
        0.0,
        trainable=True,
        name="tfco_precision_at_recall_threshold",
        dtype=tf.float32,
        auto_cast=True)
  lower_bound_slack = deferred_tensor.DeferredVariable(
      0.0,
      trainable=True,
      name="tfco_precision_at_recall_slack",
      dtype=tf.float32,
      constraint=lambda tensor: tf.maximum(0.0, tensor),
      auto_cast=True)
  upper_bound_slack = deferred_tensor.DeferredVariable(
      0.0,
      trainable=True,
      name="tfco_precision_at_recall_slack",
      dtype=tf.float32,
      constraint=lambda tensor: tf.maximum(0.0, tensor),
      auto_cast=True)

  return expression.BoundedExpression(
      lower_bound=_precision_at_recall_bound(
          context,
          recall_target=recall_target,
          threshold_tensor=lower_bound_threshold,
          slack_tensor=lower_bound_slack,
          lower_bound=True,
          upper_bound=False,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss),
      upper_bound=_precision_at_recall_bound(
          context,
          recall_target=recall_target,
          threshold_tensor=upper_bound_threshold,
          slack_tensor=upper_bound_slack,
          lower_bound=False,
          upper_bound=True,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss))


def _pr_auc_bound(context, bins, include_threshold, lower_bound, upper_bound,
                  penalty_loss, constraint_loss):
  """Creates an `Expression` representing an approximate precision-recall AUC.

  The result of this function represents a Riemann approximation to the area
  under the precision-recall curve (recall on the horizontal axis, precision on
  the vertical axis), using the constraint-based method proposed by:

  > Eban, Schain, Mackey, Gordon, Rifkin and Elidan. "Scalable Learning of
  > Non-Decomposable Objectives". AISTATS 2017.

  If you're going to be lower-bounding or maximizing the result of this
  function, then need to set the lower_bound parameter to `True`. Likewise, if
  you're going to be upper-bounding or minimizing the result of this function
  (which normally wouldn't make much sense for precision-recall AUC), then the
  upper_bound parameter must be `True`. At least one of these parameters *must*
  be `True`, and it's permitted for both of them to be `True` (but we recommend
  against this, since it would result in equality constraints, which might cause
  problems during optimization and/or post-processing).

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    bins: positive integer, the number of "rectangles" to use for the Riemann
      approximation to precision-recall AUC.
    include_threshold: if False, the thresholds associated with every bin in the
      Riemann approximation will be constrained to sum to zero. In other words,
      we'll remove one degree of freedom, causing there to be effectively bins-1
      thresholds, instead of bins thresholds.
    lower_bound: bool, `True` if you want the result of this function to
      lower-bound the approximate precision-recall AUC.
    upper_bound: bool, `True` if you want the result of this function to
      upper-bound the approximate precision-recall AUC.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate. This loss must be "normalized" (see
      `BinaryClassificationLoss.is_normalized`).

  Returns:
    An `Expression` representing a Riemann approximation to precision-recall
    AUC.

  Raises:
    TypeError: if the context is not a SubsettableContext, the number of bins is
      not an integer, or either loss is not a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels, the number of bins is
      non-positive, both lower_bound and upper_bound are `False`, or the
      constraint_loss is not normalized.
  """
  if not isinstance(bins, numbers.Integral):
    raise TypeError("number of pr_auc bins must be an integer")
  if bins <= 0:
    raise ValueError("number of pr_auc bins must be strictly positive")

  constraint = None
  if not include_threshold:
    # If include_threshold is False, then we center the thresholds around zero,
    # effectively causing there to be no overall threshold.
    constraint = lambda tensor: tensor - tf.reduce_mean(tensor)

  # Ideally the thresholds and slack variables would have the same dtype as the
  # predictions, but we might not know their dtype (e.g. in eager mode), so
  # instead we always use float32 with auto_cast=True.
  thresholds = deferred_tensor.DeferredVariable(
      np.zeros((bins,)),
      trainable=True,
      name="tfco_pr_auc_thresholds",
      dtype=tf.float32,
      constraint=constraint,
      auto_cast=True)
  slacks = deferred_tensor.DeferredVariable(
      np.zeros((bins,)),
      trainable=True,
      name="tfco_pr_auc_slacks",
      dtype=tf.float32,
      constraint=lambda tensor: tf.maximum(0.0, tensor),
      auto_cast=True)

  average_precision_expression = None
  for bin_index in xrange(bins):
    recall_target = (bin_index + 0.5) / bins
    precision_expression = _precision_at_recall_bound(
        context,
        recall_target=recall_target,
        threshold_tensor=thresholds[bin_index],
        slack_tensor=slacks[bin_index],
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)
    if average_precision_expression is None:
      average_precision_expression = precision_expression
    else:
      average_precision_expression += precision_expression

  return average_precision_expression / bins


def pr_auc(context,
           bins,
           include_threshold=True,
           penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
           constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing an approximate precision-recall AUC.

  The result of this function represents a Riemann approximation to the area
  under the precision-recall curve (recall on the horizontal axis, precision on
  the vertical axis), using the constraint-based method proposed by:

  > Eban, Schain, Mackey, Gordon, Rifkin and Elidan. "Scalable Learning of
  > Non-Decomposable Objectives". AISTATS 2017.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. This context *must* contain labels.
    bins: positive integer, the number of "rectangles" to use for the Riemann
      approximation to precision-recall AUC.
    include_threshold: if False, the thresholds associated with every bin in the
      Riemann approximation will be constrained to sum to zero. In other words,
      we'll remove one degree of freedom, causing there to be effectively bins-1
      thresholds, instead of bins thresholds.
    penalty_loss: `BinaryClassificationLoss`, the (differentiable) loss function
      to use when calculating the "penalty" approximation to the rate.
    constraint_loss: `BinaryClassificationLoss`, the (not necessarily
      differentiable) loss function to use when calculating the "constraint"
      approximation to the rate. This loss must be "normalized" (see
      `BinaryClassificationLoss.is_normalized`).

  Returns:
    An `Expression` representing a Riemann approximation to precision-recall
    AUC.

  Raises:
    TypeError: if the context is not a SubsettableContext, the number of bins is
      not an integer, or either loss is not a BinaryClassificationLoss.
    ValueError: if the context doesn't contain labels, the number of bins is
      non-positive, or the constraint_loss is not normalized.
  """
  return expression.BoundedExpression(
      lower_bound=_pr_auc_bound(
          context,
          bins,
          include_threshold=include_threshold,
          lower_bound=True,
          upper_bound=False,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss),
      upper_bound=_pr_auc_bound(
          context,
          bins,
          include_threshold=include_threshold,
          lower_bound=False,
          upper_bound=True,
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss))
