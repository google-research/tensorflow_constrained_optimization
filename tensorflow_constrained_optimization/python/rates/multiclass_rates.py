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
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
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
                                1{z[i, j] > z[i, k] for all k != j}}
        / sum_i{w_i * d_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), w_i is the given
  weights, and c_i and d_i are indicators for which examples to include in the
  numerator and denominator (all four of z, w, c and d are in the contexts).

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
    total_rate := sum_i{w_i * c_i * sum_k{
          matrix[argmax_j z[i, j], k] * y[i, k]
        }} / sum_i{w_i * d_i}
  in words, it's the average (notice the denominator) over all examples (i) of
  the cost matrix[j, k] where k is the true label and j is the prediction.
  Here, z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i and d_i are indicators for which examples to
  include in the numerator and denominator (all five of z, y, w, c and d are in
  the context).

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
    raise ValueError("this rate requires a context with labels")
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
    positive_rate := sum_i{w_i * c_i * positive_class[argmax_j z[i, j]]}
        / sum_i{w_i * c_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), w_i is the per-example
  weights, and c_i is an indicator for which examples to include in the rate
  (all three of z, w and c are in the context). Additionally, positive_class[j]
  is the probability that the jth class should be treated as "positive".

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
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
    negative_rate := sum_i{w_i * c_i * (1 - positive_class[argmax_j z[i, j]])}
        / sum_i{w_i * c_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), w_i is the per-example
  weights, and c_i is an indicator for which examples to include in the rate
  (all three of z, w and c are in the context). Additionally, positive_class[j]
  is the probability that the jth class should be treated as "positive".

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
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
    error_rate := sum_i{w_i * c_i * sum_j{y[i, j] * 1{j != argmax_j' z[i, j']}}
        / sum_i{w_i * c_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to
  include in the numerator and denominator (all four of z, y, w and c are in
  the context).

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
    error_rate := sum_i{w_i * c_i * y[i, argmax_j z[i, j]]}
        / sum_i{w_i * c_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to
  include in the numerator and denominator (all four of z, y, w and c are in
  the context).

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


def true_positive_rate(context,
                       positive_class,
                       penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                       constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass true positive rate.

  The result of this function represents:
    true_positive_rate := sum_i{w_i * c_i *
          sum_j{y[i, j] * positive_class[j]} * positive_class[argmax_j z[i, j]]
        } / sum_i{w_i * c_i * sum_j{y[i, j] * positive_class[j]}}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to include
  in the numerator and denominator (all four of z, y, w and c are in the
  context). Additionally, positive_class[j] is the probability that the jth
  class should be treated as "positive".

  Unlike the "true_positive_proportion", which is the number of
  positively-labeled examples on which we make a positive prediction, divided by
  the total number of examples, this function instead divides only by the number
  of *positively-labeled* examples.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing true_positive_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context does not contain labels, or is not a multiclass
      context, or positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("true_positive_rate requires a context with labels")

  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  def positive_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), positive_class, axes=(1, 0))

  # Notice that context.subset() handles floating-point arguments as
  # probabilities (which is what we want, if the labels are themselves
  # non-boolean, and are therefore considered probabilities).
  positive_context = context.subset(
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.constraint_labels),
  )

  return _unlabeled_rate(
      numerator_context=positive_context,
      denominator_context=positive_context,
      coefficients=positive_class,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_negative_rate(context,
                        positive_class,
                        penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                        constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass false negative rate.

  The result of this function represents:
    false_negative_rate := sum_i{w_i * c_i *
          sum_j{y[i, j] * positive_class[j]} *
          (1 - positive_class[argmax_j z[i, j]])
        } / sum_i{w_i * c_i * sum_j{y[i, j] * positive_class[j]}}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to include
  in the numerator and denominator (all four of z, y, w and c are in the
  context). Additionally, positive_class[j] is the probability that the jth
  class should be treated as "positive".

  Unlike the "false_negative_proportion", which is the number of
  positively-labeled examples on which we make a negative prediction, divided by
  the total number of examples, this function instead divides only by the number
  of *positively-labeled* examples.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing false_negative_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context does not contain labels, or is not a multiclass
      context, or positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("false_negative_rate requires a context with labels")

  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  def positive_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), positive_class, axes=(1, 0))

  # Notice that context.subset() handles floating-point arguments as
  # probabilities (which is what we want, if the labels are themselves
  # non-boolean, and are therefore considered probabilities).
  positive_context = context.subset(
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.constraint_labels),
  )

  return _unlabeled_rate(
      numerator_context=positive_context,
      denominator_context=positive_context,
      coefficients=(1.0 - positive_class),
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_positive_rate(context,
                        positive_class,
                        penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                        constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass false positive rate.

  The result of this function represents:
    false_positive_rate := sum_i{w_i * c_i *
          sum_j{y[i, j] * (1 - positive_class[j])} *
          positive_class[argmax_j z[i, j]]
        } / sum_i{w_i * c_i * sum_j{y[i, j] * (1 - positive_class[j])}}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to include
  in the numerator and denominator (all four of z, y, w and c are in the
  context). Additionally, positive_class[j] is the probability that the jth
  class should be treated as "positive".

  Unlike the "false_positive_proportion", which is the number of
  negatively-labeled examples on which we make a positive prediction, divided by
  the total number of examples, this function instead divides only by the number
  of *negatively-labeled* examples.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing false_positive_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context does not contain labels, or is not a multiclass
      context, or positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("false_positive_rate requires a context with labels")

  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  def negative_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), 1.0 - positive_class, axes=(1, 0))

  # Notice that context.subset() handles floating-point arguments as
  # probabilities (which is what we want, if the labels are themselves
  # non-boolean, and are therefore considered probabilities).
  negative_context = context.subset(
      deferred_tensor.DeferredTensor.apply(negative_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(negative_labels_fn,
                                           raw_context.constraint_labels),
  )

  return _unlabeled_rate(
      numerator_context=negative_context,
      denominator_context=negative_context,
      coefficients=positive_class,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def true_negative_rate(context,
                       positive_class,
                       penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                       constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass true negative rate.

  The result of this function represents:
    true_negative_rate := sum_i{w_i * c_i *
          sum_j{y[i, j] * (1 - positive_class[j])} *
          (1 - positive_class[argmax_j z[i, j]])
        } / sum_i{w_i * c_i * sum_j{y[i, j] * (1 - positive_class[j])}}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to include
  in the numerator and denominator (all four of z, y, w and c are in the
  context). Additionally, positive_class[j] is the probability that the jth
  class should be treated as "positive".

  Unlike the "true_negative_proportion", which is the number of
  negatively-labeled examples on which we make a negative prediction, divided by
  the total number of examples, this function instead divides only by the number
  of *negatively-labeled* examples.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing true_negative_rate (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context does not contain labels, or is not a multiclass
      context, or positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("true_negative_rate requires a context with labels")

  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  def negative_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), 1.0 - positive_class, axes=(1, 0))

  # Notice that context.subset() handles floating-point arguments as
  # probabilities (which is what we want, if the labels are themselves
  # non-boolean, and are therefore considered probabilities).
  negative_context = context.subset(
      deferred_tensor.DeferredTensor.apply(negative_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(negative_labels_fn,
                                           raw_context.constraint_labels),
  )

  return _unlabeled_rate(
      numerator_context=negative_context,
      denominator_context=negative_context,
      coefficients=(1.0 - positive_class),
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def true_positive_proportion(context,
                             positive_class,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass true positive proportion.

  The result of this function represents:
    true_positive_proportion := sum_i{w_i * c_i *
          sum_j{y[i, j] * positive_class[j]} * positive_class[argmax_j z[i, j]]
        } / sum_i{w_i * c_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to include
  in the numerator and denominator (all four of z, y, w and c are in the
  context). Additionally, positive_class[j] is the probability that the jth
  class should be treated as "positive".

  Unlike the "true_positive_rate", which is the number of positively-labeled
  examples on which we make a positive prediction, divided by the number of
  positively-labeled examples, this function instead divides by the total number
  of examples.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing true_positive_proportion (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context does not contain labels, or is not a multiclass
      context, or positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("true_positive_proportion requires a context with labels")

  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  def positive_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), positive_class, axes=(1, 0))

  # Notice that context.subset() handles floating-point arguments as
  # probabilities (which is what we want, if the labels are themselves
  # non-boolean, and are therefore considered probabilities).
  positive_context = context.subset(
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.constraint_labels),
  )

  return _unlabeled_rate(
      numerator_context=positive_context,
      denominator_context=context,
      coefficients=positive_class,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_negative_proportion(context,
                              positive_class,
                              penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                              constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass false negative proportion.

  The result of this function represents:
    false_negative_proportion := sum_i{w_i * c_i *
          sum_j{y[i, j] * positive_class[j]} *
          (1 - positive_class[argmax_j z[i, j]])
        } / sum_i{w_i * c_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to include
  in the numerator and denominator (all four of z, y, w and c are in the
  context). Additionally, positive_class[j] is the probability that the jth
  class should be treated as "positive".

  Unlike the "false_negative_rate", which is the number of positively-labeled
  examples on which we make a negative prediction, divided by the number of
  positively-labeled examples, this function instead divides by the total number
  of examples.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing false_negative_proportion (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context does not contain labels, or is not a multiclass
      context, or positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("false_negative_proportion requires a context with labels")

  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  def positive_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), positive_class, axes=(1, 0))

  # Notice that context.subset() handles floating-point arguments as
  # probabilities (which is what we want, if the labels are themselves
  # non-boolean, and are therefore considered probabilities).
  positive_context = context.subset(
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.constraint_labels),
  )

  return _unlabeled_rate(
      numerator_context=positive_context,
      denominator_context=context,
      coefficients=(1.0 - positive_class),
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_positive_proportion(context,
                              positive_class,
                              penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                              constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass false positive proportion.

  The result of this function represents:
    false_positive_proportion := sum_i{w_i * c_i *
          sum_j{y[i, j] * (1 - positive_class[j])} *
          positive_class[argmax_j z[i, j]]
        } / sum_i{w_i * c_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to include
  in the numerator and denominator (all four of z, y, w and c are in the
  context). Additionally, positive_class[j] is the probability that the jth
  class should be treated as "positive".

  Unlike the "false_positive_rate", which is the number of negatively-labeled
  examples on which we make a positive prediction, divided by the number of
  negatively-labeled examples, this function instead divides by the total number
  of examples.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing false_positive_proportion (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context does not contain labels, or is not a multiclass
      context, or positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("false_positive_proportion requires a context with labels")

  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  def negative_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), 1.0 - positive_class, axes=(1, 0))

  # Notice that context.subset() handles floating-point arguments as
  # probabilities (which is what we want, if the labels are themselves
  # non-boolean, and are therefore considered probabilities).
  negative_context = context.subset(
      deferred_tensor.DeferredTensor.apply(negative_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(negative_labels_fn,
                                           raw_context.constraint_labels),
  )

  return _unlabeled_rate(
      numerator_context=negative_context,
      denominator_context=context,
      coefficients=positive_class,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def true_negative_proportion(context,
                             positive_class,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a multiclass true negative proportion.

  The result of this function represents:
    true_negative_proportion := sum_i{w_i * c_i *
          sum_j{y[i, j] * (1 - positive_class[j])} *
          (1 - positive_class[argmax_j z[i, j]])
        } / sum_i{w_i * c_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to include
  in the numerator and denominator (all four of z, y, w and c are in the
  context). Additionally, positive_class[j] is the probability that the jth
  class should be treated as "positive".

  Unlike the "true_negative_rate", which is the number of negatively-labeled
  examples on which we make a negative prediction, divided by the number of
  negatively-labeled examples, this function instead divides by the total number
  of examples.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing true_negative_proportion (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context does not contain labels, or is not a multiclass
      context, or positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("true_negative_proportion requires a context with labels")

  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  def negative_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), 1.0 - positive_class, axes=(1, 0))

  # Notice that context.subset() handles floating-point arguments as
  # probabilities (which is what we want, if the labels are themselves
  # non-boolean, and are therefore considered probabilities).
  negative_context = context.subset(
      deferred_tensor.DeferredTensor.apply(negative_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(negative_labels_fn,
                                           raw_context.constraint_labels),
  )

  return _unlabeled_rate(
      numerator_context=negative_context,
      denominator_context=context,
      coefficients=(1.0 - positive_class),
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def precision_ratio(context,
                    positive_class,
                    penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                    constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates two `Expression`s representing multiclass precision as a ratio.

  The result of this function represents:
    numerator := sum_i{w_i * c_i *
          sum_j{y[i, j] * positive_class[j]} *
          positive_class[argmax_j z[i, j]]
        } / sum_i{w_i * c_i}
    denominator := sum_i{w_i * c_i * positive_class[argmax_j z[i, j]]}
        / sum_i{w_i * c_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to include
  in the numerator and denominator (all four of z, y, w and c are in the
  context). Additionally, positive_class[j] is the probability that the jth
  class should be treated as "positive".

  The reason for decomposing a precision as a separate numerator and denominator
  is to make it easy to set up constraints of the form (for example):

  > precision := numerator / denominator >= 0.9

  for which you can multiply through by the denominator to yield the equivalent
  constraint:

  > numerator >= 0.9 * denominator

  This latter form is something that we can straightforwardly handle.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An (`Expression`, `Expression`) pair representing the numerator and
      denominator of a precision (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context does not contain labels, or is not a multiclass
      context, or positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  raw_context = context.raw_context
  if (raw_context.penalty_labels is None or
      raw_context.constraint_labels is None):
    raise ValueError("precision requires a context with labels")

  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  def positive_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), positive_class, axes=(1, 0))

  # Notice that context.subset() handles floating-point arguments as
  # probabilities (which is what we want, if the labels are themselves
  # non-boolean, and are therefore considered probabilities).
  positive_context = context.subset(
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.constraint_labels),
  )

  numerator_expression = _unlabeled_rate(
      numerator_context=positive_context,
      denominator_context=context,
      coefficients=positive_class,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)
  denominator_expression = _unlabeled_rate(
      numerator_context=context,
      denominator_context=context,
      coefficients=positive_class,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)

  return numerator_expression, denominator_expression


def f_score_ratio(context,
                  positive_class,
                  beta=1.0,
                  penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                  constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates two `Expression`s representing multiclass F-score as a ratio.

  The result of this function represents:
    numerator := (1 + beta^2) * sum_i{w_i * c_i *
          sum_j{y[i, j] * positive_class[j]} *
          positive_class[1{j = argmax_j z[i, j]]
        } / sum_i{w_i * c_i}
    denominator := ((1 + beta^2) * sum_i{w_i * c_i *
          sum_j{y[i, j] * positive_class[j]} *
          positive_class[argmax_j z[i, j]]
        } + beta^2 * sum_i{w_i * c_i *
          sum_j{y[i, j] * positive_class[j]} *
          (1 - positive_class[argmax_j z[i, j]])
        } + sum_i{w_i * c_i *
          sum_j{y[i, j] * (1 - positive_class[j])} *
          positive_class[argmax_j z[i, j]]
        }) / sum_i{w_i * c_i}
  where z[i, j] is the prediction for the ith example on the jth class (so
  argmax_j z[i, j] is the ith example's predicted class), y[i, j] is the
  probability that the ith example is labeled to belong to the jth class, w_i is
  the per-example weights, and c_i is an indicator for which examples to include
  in the numerator and denominator (all four of z, y, w and c are in the
  context). Additionally, positive_class[j] is the probability that the jth
  class should be treated as "positive".

  The reason for decomposing an F-score as a separate numerator and denominator
  is to make it easy to set up constraints of the form (for example):

  > f_score := numerator / denominator >= 0.9

  for which you can multiply through by the denominator to yield the equivalent
  constraint:

  > numerator >= 0.9 * denominator

  This latter form is something that we can straightforwardly handle.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    positive_class: int, the index of the class to treat as "positive", *or* a
      collection of num_classes elements, where the ith element is the
      probability that the ith class should be treated as "positive".
    beta: non-negative float, the beta parameter to the F-score. If beta=0, then
      the result is precision, and if beta=1 (the default), then the result is
      the F1-score.
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An (`Expression`, `Expression`) pair representing the numerator and
      denominator of an F-score (as defined above).

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not
      a MulticlassLoss, or positive_class is a non-integer number.
    ValueError: if the context does not contain labels, or is not a multiclass
      context, or positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
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
        context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  num_classes = _get_num_classes(context)
  positive_class = _positive_class_as_array(positive_class, num_classes)

  def positive_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), positive_class, axes=(1, 0))

  def negative_labels_fn(labels):
    return tf.tensordot(
        tf.cast(labels, dtype=tf.float32), 1.0 - positive_class, axes=(1, 0))

  # Notice that context.subset() handles floating-point arguments as
  # probabilities (which is what we want, if the labels are themselves
  # non-boolean, and are therefore considered probabilities).
  positive_context = context.subset(
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(positive_labels_fn,
                                           raw_context.constraint_labels),
  )
  negative_context = context.subset(
      deferred_tensor.DeferredTensor.apply(negative_labels_fn,
                                           raw_context.penalty_labels),
      deferred_tensor.DeferredTensor.apply(negative_labels_fn,
                                           raw_context.constraint_labels),
  )

  numerator_expression = (1.0 + beta * beta) * _unlabeled_rate(
      numerator_context=positive_context,
      denominator_context=context,
      coefficients=positive_class,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)
  denominator_expression = (
      numerator_expression + (beta * beta) * _unlabeled_rate(
          numerator_context=positive_context,
          denominator_context=context,
          coefficients=(1.0 - positive_class),
          penalty_loss=penalty_loss,
          constraint_loss=constraint_loss) + _unlabeled_rate(
              numerator_context=negative_context,
              denominator_context=context,
              coefficients=positive_class,
              penalty_loss=penalty_loss,
              constraint_loss=constraint_loss))

  return numerator_expression, denominator_expression
