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

Example
=======

Suppose that "examples_tensor" and "labels_tensor" are placeholder `Tensor`s
containing training examples and their associated labels, and the "model"
function evaluates the model you're trying to train, returning a `Tensor` of
model evaluations. Further suppose that we wish to impose a fairness-style
constraint on a protected class (the "blue" examples). The following code will
create three contexts (see subsettable_context.py) representing all of the
examples, and only the "blue" examples, respectively.

>>> ctx = rate_context(model(examples_tensor), labels_tensor)
>>> blue_ctx = ctx.subset(examples_tensor[:, is_blue_idx])

Now that we have the contexts, we can create the rates. We'll try to minimize
the overall error rate, while constraining the true positive rate on the "blue"
class to be between 90% and 110% of the overall true positive rate:

>>> objective = error_rate(ctx)
>>> constraints = [
>>>     true_positive_rate(blue_ctx) >= 0.9 * true_positive_rate(ctx),
>>>     true_positive_rate(blue_ctx) <= 1.1 * true_positive_rate(ctx)
>>> ]

The error_rate() and true_positive_rate() functions are two of the
rate-constructing functions that are defined in this file. This objective and
list of constraints can then be passed on to the MinimizationProblem constructor
in rate_minimization_problem.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import expression
from tensorflow_constrained_optimization.python.rates import loss
from tensorflow_constrained_optimization.python.rates import subsettable_context
from tensorflow_constrained_optimization.python.rates import term

_DEFAULT_PENALTY_LOSS = loss.HingeLoss()
_DEFAULT_CONSTRAINT_LOSS = loss.ZeroOneLoss()


def _binary_classification_rate(positive_coefficient=0.0,
                                negative_coefficient=0.0,
                                numerator_context=None,
                                denominator_context=None,
                                penalty_loss=_DEFAULT_PENALTY_LOSS,
                                constraint_loss=_DEFAULT_CONSTRAINT_LOSS):
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

  The resulting `Expression` contains *two* different approximations to
  "total_rate". The "penalty" `BasicExpression` is an approximation to using
  penalty_loss, while the "constraint" `BasicExpression` is based on
  constraint_loss (if constraint_loss is the zero-one loss, which is the
  default, then the "constraint" expression will be exactly total_rate as
  defined above, with no approximation).

  The reason an `Expression` contains two `BasicExpression`s is that the
  "penalty" `BasicExpression` will be differentiable, while the "constraint"
  `BasicExpression` need not be. During optimization, the former will be used
  whenever we need to take gradients, and the latter otherwise.

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
                    "SubsettableContexts")
  raw_context = numerator_context.raw_context
  if denominator_context.raw_context != raw_context:
    raise ValueError("numerator and denominator contexts must be compatible")

  if not (isinstance(penalty_loss, loss.BinaryClassificationLoss) and
          isinstance(constraint_loss, loss.BinaryClassificationLoss)):
    raise TypeError("penalty and constraint losses must be "
                    "BinaryClassificationLosses")

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

  return expression.Expression(
      basic_expression.BasicExpression([penalty_term]),
      basic_expression.BasicExpression([constraint_term]))


def positive_prediction_rate(context,
                             penalty_loss=_DEFAULT_PENALTY_LOSS,
                             constraint_loss=_DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing a positive prediction rate.

  The result of this function represents:
    positive_rate := sum_i{w_i * c_i * 1{z_i > 0}} / sum_i{w_i * c_i}
  where z_i and w_i are the given predictions and weights, and c_i is an
  indicator for which examples to include in the rate (all three of z, w and c
  are in the context).

  Please see the documentation of _binary_classification_rate() for further
  details.

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
                             penalty_loss=_DEFAULT_PENALTY_LOSS,
                             constraint_loss=_DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing a negative prediction rate.

  The result of this function represents:
    negative_rate := sum_i{w_i * c_i * 1{z_i <= 0}} / sum_i{w_i * c_i}
  where z_i and w_i are the given predictions and weights, and c_i is an
  indicator for which examples to include in the rate (all three of z, w and c
  are in the context).

  Please see the documentation of _binary_classification_rate() for further
  details.

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
               penalty_loss=_DEFAULT_PENALTY_LOSS,
               constraint_loss=_DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing an error rate.

  The result of this function represents:
    error_rate := (
        sum_i{w_i * c_i * 1{y_i <= 0} * 1{z_i > 0}} +
        sum_i{w_i * c_i * 1{y_i > 0} * 1{z_i <= 0}}
    ) / sum_i{w_i * c_i}
  where z_i, y_i and w_i are the given predictions, labels and weights, and c_i
  is an indicator for which examples to include in the rate (all four of z, y, w
  and c are in the context).

  Please see the documentation of _binary_classification_rate() for further
  details.

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
                  penalty_loss=_DEFAULT_PENALTY_LOSS,
                  constraint_loss=_DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing an accuracy rate.

  The result of this function represents:
    accuracy_rate := (
        sum_i{w_i * c_i * 1{y_i > 0} * 1{z_i > 0}} +
        sum_i{w_i * c_i * 1{y_i <= 0} * 1{z_i <= 0}}
    ) / sum_i{w_i * c_i}
  where z_i, y_i and w_i are the given predictions, labels and weights, and c_i
  is an indicator for which examples to include in the rate (all four of z, y, w
  and c are in the context).

  Please see the documentation of _binary_classification_rate() for further
  details.

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
                       penalty_loss=_DEFAULT_PENALTY_LOSS,
                       constraint_loss=_DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing a true positive rate.

  The result of this function represents:
    true_positive_rate :=
        sum_i{w_i * c_i * 1{y_i > 0} * 1{z_i > 0}}
            / sum_i{w_i * c_i * 1{y_i > 0}}
  where z_i, y_i and w_i are the given predictions, labels and weights, and c_i
  is an indicator for which examples to include in the rate (all four of z, y, w
  and c are in the context).

  Please see the documentation of _binary_classification_rate() for further
  details.

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


# The "true positive rate" is the same thing as the "recall", so we allow it to
# be accessed by either name.
recall = true_positive_rate


def false_negative_rate(context,
                        penalty_loss=_DEFAULT_PENALTY_LOSS,
                        constraint_loss=_DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing a false negative rate.

  The result of this function represents:
    false_negative_rate :=
        sum_i{w_i * c_i * 1{y_i > 0} * 1{z_i <= 0}}
            / sum_i{w_i * c_i * 1{y_i > 0}}
  where z_i, y_i and w_i are the given predictions, labels and weights, and c_i
  is an indicator for which examples to include in the rate (all four of z, y, w
  and c are in the context).

  Please see the documentation of _binary_classification_rate() for further
  details.

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
                        penalty_loss=_DEFAULT_PENALTY_LOSS,
                        constraint_loss=_DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing a false positive rate.

  The result of this function represents:
    false_positive_rate :=
        sum_i{w_i * c_i * 1{y_i <= 0} * 1{z_i > 0}}
            / sum_i{w_i * c_i * 1{y_i <= 0}}
  where z_i, y_i and w_i are the given predictions, labels and weights, and c_i
  is an indicator for which examples to include in the rate (all four of z, y, w
  and c are in the context).

  Please see the documentation of _binary_classification_rate() for further
  details.

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
                       penalty_loss=_DEFAULT_PENALTY_LOSS,
                       constraint_loss=_DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` representing a true negative rate.

  The result of this function represents:
    true_negative_rate :=
        sum_i{w_i * c_i * 1{y_i <= 0} * 1{z_i <= 0}}
            / sum_i{w_i * c_i * 1{y_i <= 0}}
  where z_i, y_i and w_i are the given predictions, labels and weights, and c_i
  is an indicator for which examples to include in the rate (all four of z, y, w
  and c are in the context).

  Please see the documentation of _binary_classification_rate() for further
  details.

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
