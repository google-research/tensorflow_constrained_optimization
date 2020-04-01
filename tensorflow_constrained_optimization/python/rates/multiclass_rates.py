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
"""Contains functions for constructing multiclass rate expressions.

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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numbers
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import expression
from tensorflow_constrained_optimization.python.rates import loss
from tensorflow_constrained_optimization.python.rates import predicate
from tensorflow_constrained_optimization.python.rates import subsettable_context
from tensorflow_constrained_optimization.python.rates import term


def _get_num_classes(context):
  """Extracts the number of classes from a multiclass context."""
  if not isinstance(context, subsettable_context.SubsettableContext):
    raise TypeError("context must be a SubsettableContext object")
  raw_context = context.raw_context
  num_classes = raw_context.num_classes
  if num_classes is None:
    raise ValueError("multiclass rates can only be constructed from "
                     "multiclass contexts")
  return num_classes


def _positive_class_as_array(positive_class, num_classes):
  """Converts the positive_class parameter to a shape-(num_classes,) array."""
  if isinstance(positive_class, numbers.Number):
    if not isinstance(positive_class, numbers.Integral):
      raise TypeError("if positive_class is a number, it must be an integer")
    if positive_class < 0 or positive_class >= num_classes:
      raise ValueError("positive_class must be between 0 and %d (inclusive), "
                       "not %d" % (num_classes - 1, positive_class))
    result = np.zeros(num_classes, dtype=np.float32)
    result[positive_class] = 1.0
  elif isinstance(positive_class, collections.Iterable):
    result = np.array(positive_class, dtype=np.float32)
    if len(np.shape(result)) != 1:
      raise ValueError("positive_class cannot be multi-dimensional")
    if len(result) != num_classes:
      raise ValueError("if positive_class is a collection, it must contain "
                       "%d elements, not %d" % (num_classes, len(result)))
  else:
    raise ValueError("positive_class must be either an integer, or a "
                     "collection with num_classes elements, when creating a "
                     "rate from a multiclass context")
  return result


def _unlabeled_rate(numerator_context,
                    denominator_context,
                    coefficients,
                    penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                    constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass rate on an unlabeled dataset.

  The result of this function represents:
    total_rate := sum_j coefficient[j] * prediction_rate[j]
  where predictions_rate[j] represents the rate at which the jth class is
  predicted:
    prediction_rate[j] := sum_i{w_i * c_i *
                                1{z[i, j] > z[i, k] forall k != j}}
        / sum_i{w_i * d_i}
  where z[i, j] is the prediction for the ith example on the jth class, w_i is
  the given weights, and c_i and d_i are indicators for which examples to
  include in the numerator and denominator (all four of z, w, c and d are in the
  contexts).

  Args:
    numerator_context: multiclass `SubsettableContext`, the block of data to use
      when calculating the numerators of the rates.
    denominator_context: multiclass `SubsettableContext`, the block of data to
      use when calculating the denominators of the rates.
    coefficients: a list of num_classes coefficients.
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rates.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rates.

  Returns:
    An `Expression` representing total_rate (as defined above).

  Raises:
    TypeError: if either context is not a SubsettableContext, or either loss is
      not a MulticlassLoss.
    ValueError: if the two contexts are incompatible, or either is not a
      multiclass context, or the coefficients argument contains the wrong number
      of coefficients.
  """
  if not (
      isinstance(numerator_context, subsettable_context.SubsettableContext) and
      isinstance(denominator_context, subsettable_context.SubsettableContext)):
    raise TypeError("numerator and denominator contexts must be "
                    "SubsettableContext objects")
  raw_context = numerator_context.raw_context
  if denominator_context.raw_context != raw_context:
    raise ValueError("numerator and denominator contexts must be compatible")
  num_classes = raw_context.num_classes
  if num_classes is None:
    raise ValueError("multiclass rates can only be constructed from "
                     "multiclass contexts")

  if not (isinstance(penalty_loss, loss.MulticlassLoss) and
          isinstance(constraint_loss, loss.MulticlassLoss)):
    raise TypeError("penalty and constraint losses must be MulticlassLoss "
                    "objects")

  # The ith coefficient is the cost associated with predicting class i.
  if len(coefficients) != num_classes:
    raise ValueError("multiclass cost coefficients must be length %d (it is "
                     "length %d)" % (num_classes, len(coefficients)))

  penalty_term = term.MulticlassTerm.ratio(
      coefficients=coefficients,
      predictions=raw_context.penalty_predictions,
      weights=raw_context.penalty_weights,
      numerator_predicate=numerator_context.penalty_predicate,
      denominator_predicate=denominator_context.penalty_predicate,
      loss=penalty_loss)
  constraint_term = term.MulticlassTerm.ratio(
      coefficients=coefficients,
      predictions=raw_context.constraint_predictions,
      weights=raw_context.constraint_weights,
      numerator_predicate=numerator_context.constraint_predicate,
      denominator_predicate=denominator_context.constraint_predicate,
      loss=constraint_loss)

  return expression.ExplicitExpression(
      basic_expression.BasicExpression([penalty_term]),
      basic_expression.BasicExpression([constraint_term]))


def _labeled_rate(numerator_context,
                  denominator_context,
                  matrix,
                  penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                  constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass rate on a labeled dataset.

  The result of this function represents:
    total_rate := sum_i{sum_j{sum_k{w_i * c_i * matrix[j, k] *
                                    1{l_i = k} *
                                    1{z[i, j] > z[i, k] forall k != j}}}} /
        / sum_i{w_i * d_i}
  in words, it's the average (notice the denominator) over all examples (i) of
  the cost matrix[j, k] where k is the true label and j is the prediction.
  Here, z[i, j] is the prediction for the ith example on the jth class, l_i is
  the given labels, w_i is the given weights, and c_i and d_i are indicators for
  which examples to include in the numerator and denominator (all five of z, l,
  w, c and d are in the context).

  Args:
    numerator_context: multiclass `SubsettableContext`, the block of data to use
      when calculating the numerators of the rates.
    denominator_context: multiclass `SubsettableContext`, the block of data to
      use when calculating the denominators of the rates.
    matrix: a matrix of shape (num_classes, num_classes) of coefficients, for
      which matrix[j, k] represents the cost of predicting class j on an example
      labeled as belonging to class k.
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rates.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rates.

  Returns:
    An `Expression` representing total_rate (as defined above).

  Raises:
    TypeError: if either context is not a SubsettableContext, or either loss is
      not a MulticlassLoss.
    ValueError: if the two contexts are incompatible, or either is unlabeled, or
      is not a multiclass context, or the matrix contains the wrong number of
      coefficients.
  """
  if not (
      isinstance(numerator_context, subsettable_context.SubsettableContext) and
      isinstance(denominator_context, subsettable_context.SubsettableContext)):
    raise TypeError("numerator and denominator contexts must be "
                    "SubsettableContext objects")
  raw_context = numerator_context.raw_context
  if denominator_context.raw_context != raw_context:
    raise ValueError("numerator and denominator contexts must be compatible")
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("this multiclass_rate requires a context with labels")
  num_classes = raw_context.num_classes
  if num_classes is None:
    raise ValueError("multiclass rates can only be constructed from "
                     "multiclass contexts")

  if not (isinstance(penalty_loss, loss.MulticlassLoss) and
          isinstance(constraint_loss, loss.MulticlassLoss)):
    raise TypeError("penalty and constraint losses must be MulticlassLoss "
                    "objects")

  # The labels are associated with the columns, and the predictions with the
  # rows. In other words, matrix[j, k] is the cost associated with predicting
  # class j on an example labeled as class k.
  matrix_shape = np.shape(matrix)
  if len(matrix_shape) != 2:
    raise ValueError("multiclass cost matrix must be two-dimensional (it is "
                     "%d-dimensional)" % len(matrix_shape))
  if matrix_shape != (num_classes, num_classes):
    raise ValueError(
        "multiclass cost matrix must be %d*%d (it is %d*%d)" %
        (num_classes, num_classes, matrix_shape[0], matrix_shape[1]))

  penalty_terms = []
  constraint_terms = []
  for ii in xrange(num_classes):
    coefficients = matrix[:, ii]

    # Each set of labels is assumed to be a distribution (e.g. a one-hot
    # encoding), so we stick them into the numerator predicate.
    penalty_numerator_predicate = (
        numerator_context.penalty_predicate
        & predicate.Predicate(raw_context.penalty_labels[:, ii]))
    constraint_numerator_predicate = (
        numerator_context.constraint_predicate
        & predicate.Predicate(raw_context.constraint_labels[:, ii]))

    penalty_terms.append(
        term.MulticlassTerm.ratio(
            coefficients=coefficients,
            predictions=raw_context.penalty_predictions,
            weights=raw_context.penalty_weights,
            numerator_predicate=penalty_numerator_predicate,
            denominator_predicate=denominator_context.penalty_predicate,
            loss=penalty_loss))
    constraint_terms.append(
        term.MulticlassTerm.ratio(
            coefficients=coefficients,
            predictions=raw_context.constraint_predictions,
            weights=raw_context.constraint_weights,
            numerator_predicate=constraint_numerator_predicate,
            denominator_predicate=denominator_context.constraint_predicate,
            loss=constraint_loss))

  return expression.ExplicitExpression(
      basic_expression.BasicExpression(penalty_terms),
      basic_expression.BasicExpression(constraint_terms))


def positive_prediction_rate(context,
                             positive_class,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass positive prediction rate.

  The result of this function represents:
    positive_rate := sum_i{w_i * c_i *
        1{z[i, positive_class] > z[i, k] forall k != positive_class}}
        / sum_i{w_i * c_i}
  where z_i and w_i are the given predictions and weights, and c_i is an
  indicator for which examples to include in the rate (all three of z, w and c
  are in the context).

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element indicates
      whether the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing positive_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context is not a multiclass context, or positive_class
      is an integer outside the range [0,num_classes), or is a collection not
      containing num_classes elements.
  """
  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  return _unlabeled_rate(
      numerator_context=context,
      denominator_context=context,
      coefficients=positive_class,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def negative_prediction_rate(context,
                             positive_class,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass negative prediction rate.

  The result of this function represents:
    negative_rate := sum_i{w_i * c_i *
        1{exists j != positive_class s.t. z[i, j] > z[i, positive_class]}}
        / sum_i{w_i * c_i}
  where z_i and w_i are the given predictions and weights, and c_i is an
  indicator for which examples to include in the rate (all three of z, w and c
  are in the context).

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element indicates
      whether the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing negative_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context is not a multiclass context, or positive_class
      is an integer outside the range [0,num_classes), or is a collection not
      containing num_classes elements.
  """
  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  return _unlabeled_rate(
      numerator_context=context,
      denominator_context=context,
      coefficients=(1.0 - positive_class),
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def error_rate(context,
               penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
               constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass error rate.

  The result of this function represents:
    error_rate := sum_i{w_i * c_i * 1{exists j != y_i s.t. z[i, j] > z[i, y_i]}}
        / sum_i{w_i * c_i}
  where z_i, y_i and w_i are the given predictions, labels and weights, and c_i
  is an indicator for which examples to include in the rate (all four of z, y, w
  and c are in the context).

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate. This context *must* contain labels.
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing error_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a MulticlassLoss.
    ValueError: if the context doesn't contain labels, or is not a multiclass
      context.
  """
  num_classes = _get_num_classes(context)
  return _labeled_rate(
      numerator_context=context,
      denominator_context=context,
      matrix=~np.eye(num_classes, dtype=np.bool),
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def accuracy_rate(context,
                  penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                  constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass accuracy rate.

  The result of this function represents:
    accuracy_rate := sum_i{w_i * c_i * 1{z[i, y_i] > z[i, j] forall j != y_i}}
        / sum_i{w_i * c_i}
  where z_i, y_i and w_i are the given predictions, labels and weights, and c_i
  is an indicator for which examples to include in the rate (all four of z, y, w
  and c are in the context).

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate. This context *must* contain labels.
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` for accuracy_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, or either loss is not
      a MulticlassLoss.
    ValueError: if the context doesn't contain labels, or is not a multiclass
      context.
  """
  num_classes = _get_num_classes(context)
  return _labeled_rate(
      numerator_context=context,
      denominator_context=context,
      matrix=np.eye(num_classes, dtype=np.bool),
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)
