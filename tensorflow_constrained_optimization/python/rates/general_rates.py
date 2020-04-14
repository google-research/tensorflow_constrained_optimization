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
"""Contains functions for constructing binary or multiclass rate expressions.

There are a number of rates (e.g. error_rate()) that can be defined for either
binary classification or multiclass contexts. The former rates are implemented
in binary_rates.py, and the latter in multiclass_rates.py. In this file, the
given functions choose which rate to create based on the type of the context:
for multiclass contexts, they'll call the corresponding implementation in
multiclass_rates.py, otherwise, they'll call binary_rates.py.

Many of the functions in this file take the optional "positive_class" parameter,
which tells us which classes should be considered "positive" (for e.g. the
positive prediction rate). This parameter *must* be provided for multiclass
contexts, and must *not* be provided for non-multiclass contexts.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_constrained_optimization.python.rates import binary_rates
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import multiclass_rates
from tensorflow_constrained_optimization.python.rates import subsettable_context


def _is_multiclass(context):
  """Returns True iff we're given a multiclass context."""
  if not isinstance(context, subsettable_context.SubsettableContext):
    raise TypeError("context must be a SubsettableContext object")
  raw_context = context.raw_context
  return raw_context.num_classes is not None


def positive_prediction_rate(context,
                             positive_class=None,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a positive prediction rate.

  A positive prediction rate is the number of examples within the given context
  on which the model makes a positive prediction, divided by the number of
  examples within the context. For multiclass problems, the positive_class
  argument, which tells us which class (or classes) should be treated as
  positive, must also be provided.

  Please see the docstrings of positive_prediction_rate() in binary_rates.py and
  multiclass_rates.py for further details.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. If this is a multiclass context, we'll calculate the multiclass
      version of the rate.
    positive_class: None for a non-multiclass problem. Otherwise, an int, the
      index of the class to treat as "positive", *or* a collection of
      num_classes elements, where the ith element is the probability that the
      ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing the positive prediction rate.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if positive_class is provided for a non-multiclass context, or
      is *not* provided for a multiclass context. In the latter case, an error
      will also be raised if positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.positive_prediction_rate(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError("positive_class cannot be provided to "
                     "positive_prediction_rate unless it's also given a "
                     "multiclass context")
  return binary_rates.positive_prediction_rate(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def negative_prediction_rate(context,
                             positive_class=None,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a negative prediction rate.

  A negative prediction rate is the number of examples within the given context
  on which the model makes a negative prediction, divided by the number of
  examples within the context. For multiclass problems, the positive_class
  argument, which tells us which class (or classes) should be treated as
  positive, must also be provided.

  Please see the docstrings of negative_prediction_rate() in binary_rates.py and
  multiclass_rates.py for further details.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. If this is a multiclass context, we'll calculate the multiclass
      version of the rate.
    positive_class: None for a non-multiclass problem. Otherwise, an int, the
      index of the class to treat as "positive", *or* a collection of
      num_classes elements, where the ith element is the probability that the
      ith class should be treated as "positive".
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing the negative prediction rate.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if positive_class is provided for a non-multiclass context, or
      is *not* provided for a multiclass context. In the latter case, an error
      will also be raised if positive_class is an integer outside the range
      [0,num_classes), or is a collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.negative_prediction_rate(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError("positive_class cannot be provided to "
                     "negative_prediction_rate unless it's also given a "
                     "multiclass context")
  return binary_rates.negative_prediction_rate(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def error_rate(context,
               penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
               constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for an error rate.

  An error rate is the number of examples within the given context on which the
  model makes an incorrect prediction, divided by the number of examples within
  the context.

  Please see the docstrings of error_rate() in binary_rates.py and
  multiclass_rates.py for further details.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. If this is a multiclass context, we'll calculate the multiclass
      version of the rate.
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing the error rate.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass).
    ValueError: if the context doesn't contain labels.
  """
  if _is_multiclass(context):
    return multiclass_rates.error_rate(
        context=context,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  return binary_rates.error_rate(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def accuracy_rate(context,
                  penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                  constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for an accuracy rate.

  An accuracy rate is the number of examples within the given context on which
  the model makes a correct prediction, divided by the number of examples within
  the context.

  Please see the docstrings of accuracy_rate() in binary_rates.py and
  multiclass_rates.py for further details.

  Args:
    context: `SubsettableContext`, the block of data to use when calculating the
      rate. If this is a multiclass context, we'll calculate the multiclass
      version of the rate.
    penalty_loss: `MulticlassLoss`, the (differentiable) loss function to use
      when calculating the "penalty" approximation to the rate.
    constraint_loss: `MulticlassLoss`, the (not necessarily differentiable) loss
      function to use when calculating the "constraint" approximation to the
      rate.

  Returns:
    An `Expression` representing the accuracy rate.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass).
    ValueError: if the context doesn't contain labels.
  """
  if _is_multiclass(context):
    return multiclass_rates.accuracy_rate(
        context=context,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  return binary_rates.accuracy_rate(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)
