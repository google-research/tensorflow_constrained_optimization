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

import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import basic_expression
from tensorflow_constrained_optimization.python.rates import binary_rates
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import expression
from tensorflow_constrained_optimization.python.rates import multiclass_rates
from tensorflow_constrained_optimization.python.rates import subsettable_context
from tensorflow_constrained_optimization.python.rates import term


def _is_multiclass(context):
  """Returns True iff we're given a multiclass context."""
  if not isinstance(context, subsettable_context.SubsettableContext):
    raise TypeError("context must be a SubsettableContext object")
  raw_context = context.raw_context
  return raw_context.num_classes is not None


def _ratio_bound(numerator_expression, denominator_expression, lower_bound,
                 upper_bound):
  """Creates an `Expression` for a bound on a ratio.

  The result of this function is an `Expression` representing:
    numerator / denominator_bound
  where denominator_bound is a newly-created slack variable projected to satisfy
  the following (in an update op):
    denominator_lower_bound <= denominator_bound <= 1
  Additionally, the following constraint will be added if lower_bound is True:
    denominator_bound >= denominator_expression
  and/or the following if upper_bound is true:
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
    denominator_expression: `Expression`, the denominator of the ratio. The
      value of this expression must be between zero and one.
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

  # We use an "update_ops_fn" instead of a "constraint" (which we would usually
  # prefer) to perform the projection because we want to grab the denominator
  # lower bound out of the structure_memoizer.
  def update_ops_fn(denominator_bound_variable, structure_memoizer,
                    value_memoizer):
    """Projects denominator_bound onto the feasible region."""
    del value_memoizer

    denominator_bound = tf.maximum(
        structure_memoizer[defaults.DENOMINATOR_LOWER_BOUND_KEY],
        tf.minimum(1.0, denominator_bound_variable))
    return [denominator_bound_variable.assign(denominator_bound)]

  # Ideally the slack variable would have the same dtype as the predictions, but
  # we might not know their dtype (e.g. in eager mode), so instead we always use
  # float32 with auto_cast=True.
  denominator_bound = deferred_tensor.DeferredVariable(
      1.0,
      trainable=True,
      name="tfco_denominator_bound",
      dtype=tf.float32,
      update_ops_fn=update_ops_fn,
      auto_cast=True)
  denominator_bound_basic_expression = basic_expression.BasicExpression(
      [term.TensorTerm(denominator_bound)])
  denominator_bound_expression = expression.ExplicitExpression(
      penalty_expression=denominator_bound_basic_expression,
      constraint_expression=denominator_bound_basic_expression)

  extra_constraints = []
  if lower_bound:
    extra_constraints.append(
        denominator_expression <= denominator_bound_expression)
  if upper_bound:
    extra_constraints.append(
        denominator_bound_expression <= denominator_expression)

  return expression.ConstrainedExpression(
      expression=numerator_expression._positive_scalar_div(denominator_bound),  # pylint: disable=protected-access
      extra_constraints=extra_constraints)


def _ratio(numerator_expression, denominator_expression):
  """Creates an `Expression` for a ratio.

  The result of this function is an `Expression` representing:
    numerator / denominator_bound
  where denominator_bound satisfies the following:
    denominator_lower_bound <= denominator_bound <= 1
  The resulting `Expression` will include both the implicit denominator_bound
  slack variable, and implicit constraints.

  Args:
    numerator_expression: `Expression`, the numerator of the ratio.
    denominator_expression: `Expression`, the denominator of the ratio.

  Returns:
    An `Expression` representing the ratio.

  Raises:
    TypeError: if either numerator_expression or denominator_expression is not
      an `Expression`.
  """
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


def true_positive_rate(context,
                       positive_class=None,
                       penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                       constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a true positive rate.

  A true positive rate is the number of positively-labeled examples within the
  given context on which the model makes a positive prediction, divided by the
  number of positively-labeled examples within the context. For multiclass
  problems, the positive_class argument, which tells us which class (or classes)
  should be treated as positive, must also be provided.

  Please see the docstrings of true_positive_rate() in binary_rates.py and
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
    An `Expression` representing the true positive rate.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.true_positive_rate(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError("positive_class cannot be provided to "
                     "true_positive_rate unless it's also given a multiclass "
                     "context")
  return binary_rates.true_positive_rate(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_negative_rate(context,
                        positive_class=None,
                        penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                        constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a false negative rate.

  A false negative rate is the number of positively-labeled examples within the
  given context on which the model makes a negative prediction, divided by the
  number of positively-labeled examples within the context. For multiclass
  problems, the positive_class argument, which tells us which class (or classes)
  should be treated as positive, must also be provided.

  Please see the docstrings of false_negative_rate() in binary_rates.py and
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
    An `Expression` representing the false negative rate.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.false_negative_rate(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError("positive_class cannot be provided to "
                     "false_negative_rate unless it's also given a multiclass "
                     "context")
  return binary_rates.false_negative_rate(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_positive_rate(context,
                        positive_class=None,
                        penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                        constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a false positive rate.

  A false positive rate is the number of negatively-labeled examples within the
  given context on which the model makes a positive prediction, divided by the
  number of negatively-labeled examples within the context. For multiclass
  problems, the positive_class argument, which tells us which class (or classes)
  should be treated as positive, must also be provided.

  Please see the docstrings of false_positive_rate() in binary_rates.py and
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
    An `Expression` representing the false positive rate.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.false_positive_rate(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError("positive_class cannot be provided to "
                     "false_positive_rate unless it's also given a multiclass "
                     "context")
  return binary_rates.false_positive_rate(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def true_negative_rate(context,
                       positive_class=None,
                       penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                       constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a true negative rate.

  A true negative rate is the number of negatively-labeled examples within the
  given context on which the model makes a negative prediction, divided by the
  number of negatively-labeled examples within the context. For multiclass
  problems, the positive_class argument, which tells us which class (or classes)
  should be treated as positive, must also be provided.

  Please see the docstrings of true_negative_rate() in binary_rates.py and
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
    An `Expression` representing the true negative rate.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.true_negative_rate(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError("positive_class cannot be provided to "
                     "true_negative_rate unless it's also given a multiclass "
                     "context")
  return binary_rates.true_negative_rate(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def true_positive_proportion(context,
                             positive_class=None,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a true positive proportion.

  A true positive proportion is the number of positively-labeled examples within
  the given context on which the model makes a positive prediction, divided by
  the total number of examples within the context. For multiclass problems, the
  positive_class argument, which tells us which class (or classes) should be
  treated as positive, must also be provided.

  Please see the docstrings of true_positive_proportion() in binary_rates.py and
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
    An `Expression` representing the true positive proportion.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.true_positive_proportion(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError(
        "positive_class cannot be provided to "
        "true_positive_proportion unless it's also given a multiclass "
        "context")
  return binary_rates.true_positive_proportion(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_negative_proportion(context,
                              positive_class=None,
                              penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                              constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a false negative proportion.

  A false negative proportion is the number of positively-labeled examples
  within the given context on which the model makes a negative prediction,
  divided by the total number of examples within the context. For multiclass
  problems, the positive_class argument, which tells us which class (or classes)
  should be treated as positive, must also be provided.

  Please see the docstrings of false_negative_proportion() in binary_rates.py
  and multiclass_rates.py for further details.

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
    An `Expression` representing the false negative proportion.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.false_negative_proportion(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError(
        "positive_class cannot be provided to "
        "false_negative_proportion unless it's also given a multiclass "
        "context")
  return binary_rates.false_negative_proportion(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def false_positive_proportion(context,
                              positive_class=None,
                              penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                              constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a false positive proportion.

  A false positive proportion is the number of negatively-labeled examples
  within the given context on which the model makes a positive prediction,
  divided by the total number of examples within the context. For multiclass
  problems, the positive_class argument, which tells us which class (or classes)
  should be treated as positive, must also be provided.

  Please see the docstrings of false_positive_proportion() in binary_rates.py
  and multiclass_rates.py for further details.

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
    An `Expression` representing the false positive proportion.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.false_positive_proportion(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError(
        "positive_class cannot be provided to "
        "false_positive_proportion unless it's also given a multiclass "
        "context")
  return binary_rates.false_positive_proportion(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def true_negative_proportion(context,
                             positive_class=None,
                             penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                             constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for a true negative proportion.

  A true negative proportion is the number of negatively-labeled examples within
  the given context on which the model makes a negative prediction, divided by
  the total number of examples within the context. For multiclass problems, the
  positive_class argument, which tells us which class (or classes) should be
  treated as positive, must also be provided.

  Please see the docstrings of true_negative_proportion() in binary_rates.py
  and multiclass_rates.py for further details.

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
    An `Expression` representing the true negative proportion.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.true_negative_proportion(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError(
        "positive_class cannot be provided to "
        "true_negative_proportion unless it's also given a multiclass "
        "context")
  return binary_rates.true_negative_proportion(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def precision_ratio(context,
                    positive_class=None,
                    penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                    constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates two `Expression`s representing precision as a ratio.

  A precision is the number of positively-labeled examples within the given
  context on which the model makes a positive prediction, divided by the number
  of examples within the context on which the model makes a positive prediction.
  For multiclass problems, the positive_class argument, which tells us which
  class (or classes) should be treated as positive, must also be provided.

  Please see the docstrings of precision_ratio() in binary_rates.py and
  multiclass_rates.py for further details.

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
    An (`Expression`, `Expression`) pair representing the numerator and
      denominator of a precision, respectively.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.precision_ratio(
        context=context,
        positive_class=positive_class,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError("positive_class cannot be provided to "
                     "precision_ratio unless it's also given a multiclass "
                     "context")
  return binary_rates.precision_ratio(
      context=context,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def precision(context,
              positive_class=None,
              penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
              constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression`s for precision.

  A precision is the number of positively-labeled examples within the given
  context on which the model makes a positive prediction, divided by the number
  of examples within the context on which the model makes a positive prediction.
  For multiclass problems, the positive_class argument, which tells us which
  class (or classes) should be treated as positive, must also be provided.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
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
    An `Expression` representing the precision.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  numerator_expression, denominator_expression = precision_ratio(
      context=context,
      positive_class=positive_class,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)
  return _ratio(
      numerator_expression=numerator_expression,
      denominator_expression=denominator_expression)


def f_score_ratio(context,
                  beta=1.0,
                  positive_class=None,
                  penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
                  constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates two `Expression`s representing F-score as a ratio.

  An F score [Wikipedia](https://en.wikipedia.org/wiki/F1_score), is a harmonic
  mean of recall and precision, where the parameter beta weights the importance
  of the precision component. If beta=1, the result is the usual harmonic mean
  (the F1 score) of these two quantities. If beta=0, the result is the
  precision, and as beta goes to infinity, the result converges to the recall.
  For multiclass problems, the positive_class argument, which tells us which
  class (or classes) should be treated as positive, must also be provided.

  Please see the docstrings of f_score_ratio() in binary_rates.py and
  multiclass_rates.py for further details.

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
    beta: non-negative float, the beta parameter to the F-score. If beta=0, then
      the result is precision, and if beta=1 (the default), then the result is
      the F1-score.
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
    An (`Expression`, `Expression`) pair representing the numerator and
      denominator of an F-score, respectively.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  if _is_multiclass(context):
    return multiclass_rates.f_score_ratio(
        context=context,
        positive_class=positive_class,
        beta=beta,
        penalty_loss=penalty_loss,
        constraint_loss=constraint_loss)

  if positive_class is not None:
    raise ValueError("positive_class cannot be provided to "
                     "f_score_ratio unless it's also given a multiclass "
                     "context")
  return binary_rates.f_score_ratio(
      context=context,
      beta=beta,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)


def f_score(context,
            beta=1.0,
            positive_class=None,
            penalty_loss=defaults.DEFAULT_PENALTY_LOSS,
            constraint_loss=defaults.DEFAULT_CONSTRAINT_LOSS):
  """Creates an `Expression` for F-score.

  An F score [Wikipedia](https://en.wikipedia.org/wiki/F1_score), is a harmonic
  mean of recall and precision, where the parameter beta weights the importance
  of the precision component. If beta=1, the result is the usual harmonic mean
  (the F1 score) of these two quantities. If beta=0, the result is the
  precision, and as beta goes to infinity, the result converges to the recall.
  For multiclass problems, the positive_class argument, which tells us which
  class (or classes) should be treated as positive, must also be provided.

  Args:
    context: multiclass `SubsettableContext`, the block of data to use when
      calculating the rate.
    beta: non-negative float, the beta parameter to the F-score. If beta=0, then
      the result is precision, and if beta=1 (the default), then the result is
      the F1-score.
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
    An `Expression` representing the F-score.

  Raises:
    TypeError: if the context is not a SubsettableContext, either loss is not a
      BinaryClassificationLoss (if the context is non-multiclass) or a
      MulticlassLoss (if the context is multiclass). In the latter case, an
      error will also be raised if positive_class is a non-integer number.
    ValueError: if the context doesn't contain labels, or positive_class is
      provided for a non-multiclass context, or is *not* provided for a
      multiclass context. In the latter case, an error will also be raised if
      positive_class is an integer outside the range [0,num_classes), or is a
      collection not containing num_classes elements.
  """
  numerator_expression, denominator_expression = f_score_ratio(
      context=context,
      beta=beta,
      positive_class=positive_class,
      penalty_loss=penalty_loss,
      constraint_loss=constraint_loss)
  return _ratio(
      numerator_expression=numerator_expression,
      denominator_expression=denominator_expression)
