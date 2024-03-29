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
"""Contains loss functions for use in rate constraints.

The "rates" used in this library represent e.g. the true positive rate, which is
the proportion of positively-labeled examples on which the model makes a
positive prediction. We cannot optimize over such rates directly, since the
indicator function for "do we make a positive prediction on this example" is
discontinuous, and hence not differentiable. Hence, we relax this indicator
function using a loss function, several alternatives for which are defined in
this file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import helpers


def _convert_to_binary_classification_predictions(predictions):
  """Converts a `Tensor` into a set of binary classification predictions.

  This function checks that the given `Tensor` is floating-point, and that it is
  trivially convertible to rank-1, i.e. has only one "nontrivial" dimension
  (e.g. the shapes [1000] and [1, 1, None, 1] are allowed, but [None, 1, None]
  and [50, 10] are not). If it satisfies these conditions, then it is reshaped
  to be rank-1 (if necessary) and returned.

  Args:
    predictions: a rank-1 floating-point `Tensor` of predictions.

  Returns:
    The predictions `Tensor`, reshaped to be rank-1, if necessary.

  Raises:
    TypeError: if "predictions" is not a floating-point `Tensor`.
    ValueError: if "predictions" is not trivially convertible to rank-1.
  """
  if not tf.is_tensor(predictions):
    raise TypeError("predictions must be a Tensor")
  if not predictions.dtype.is_floating:
    raise TypeError("predictions must be floating-point")

  return helpers.convert_to_1d_tensor(predictions, name="predictions")


@six.add_metaclass(abc.ABCMeta)
class Loss(helpers.RateObject):
  """Abstract base class for losses.

  We use `Loss`es as keys in dictionaries (see the Term.key property), so every
  `Loss` must implement the __hash__ and __eq__ methods.
  """

  @abc.abstractproperty
  def is_differentiable(self):
    """Returns true only if the associated loss is {sub,super}differentiable.

    This property is used to check that non-differentiable losses (e.g. the
    zero-one loss) aren't used in contexts in which we will try to optimize over
    them. Non-differentiable losses, however, can be *evaluated* safely.

    Subdifferentiability or superdifferentiability is enough: as long as we can
    optimize the loss using a gradient-based method, this method should return
    True.

    Returns:
      True if the loss is {sub,super}differentiable. False otherwise.
    """

  @abc.abstractmethod
  def __hash__(self):
    pass

  @abc.abstractmethod
  def __eq__(self, other):
    pass

  def __ne__(self, other):
    return not self.__eq__(other)


@six.add_metaclass(abc.ABCMeta)
class BinaryClassificationLoss(Loss):
  """Abstract class for binary classification losses."""

  @abc.abstractproperty
  def is_normalized(self):
    """Returns true only if the associated loss is normalized.

    We call a classification loss "normalized" if there exists a random variable
    Z such that, for any values of the predictions and weights:

    > loss(predictions, weights) = E[zero-one-loss(predictions + Z, weights)]

    where the expectation is taken over Z.

    Intuitively, a normalized loss can be interpreted as a smoothed zero-one
    loss (e.g. a ramp or a sigmoid), while a non-normalized loss will typically
    be some unbounded relaxation (e.g. a hinge).

    Returns:
      True if the loss is normalized. False otherwise.
    """

  @abc.abstractmethod
  def evaluate_binary_classification(self, predictions, weights):
    """Evaluates a binary classification loss on the given predictions.

    Given a rank-1 `Tensor` of predictions with shape (n,), where n is the
    number of examples, and a rank-2 `Tensor` of weights with shape (m, 2),
    where m is broadcastable to n, this method will return a `Tensor` of shape
    (n,) where the ith element *approximates*:

    ```python
    zero_one_loss[i] = weights[i, 0] * 1{predictions[i] > 0} +
      0.5 * (weights[i, 0] + weights[i, 1]) * 1{predictions[i] == 0} +
      weights[i, 1] * 1{predictions[i] < 0}
    ```

    where 1{} is an indicator function. For the zero-one loss, the result will
    equal the above quantity, while for other losses, it'll instead be an
    approximation. For convex losses, it will typically be a convex (in
    predictions) upper bound.

    You can think of weights[:, 0] as being the per-example costs associated
    with making a positive prediction, and weights[:, 1] as those for a negative
    prediction.

    Args:
      predictions: a `Tensor` of shape (n,), where n is the number of examples.
      weights: a `Tensor` of shape (m, 2), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      losses for each example.
    """


@six.add_metaclass(abc.ABCMeta)
class MulticlassLoss(BinaryClassificationLoss):
  """Abstract class for multiclass losses."""

  @abc.abstractproperty
  def is_normalized(self):
    pass

  @abc.abstractmethod
  def evaluate_multiclass(self, predictions, weights):
    """Evaluates a multiclass loss on the given binary predictions.

    Given a rank-2 `Tensor` of predictions with shape (n, k), where n is the
    number of examples and k is the number of classes, and another rank-2
    `Tensor` of weights with shape (m, k), where m is broadcastable to n, this
    method will return a `Tensor` of shape (n,) where the ith element
    *approximates*:

    ```python
    maximum_prediction[i] = max(predictions[i, :])
    maximum_weights[i] = [
      weights[i, j] for j in range(k)
      if predictions[i, j] >= maximum_prediction[i]]
    zero_one_loss[i] = sum(maximum_weights[i]) / len(maximum_weights[i])
    ```

    For the zero-one loss, the result will equal the above quantity, while for
    other losses, it'll instead be an approximation. For convex losses, it will
    typically be a convex (in predictions) upper bound.

    You can think of weights[:, k] as being the per-example costs associated
    with predicting the kth class.

    Args:
      predictions: a `Tensor` of shape (n, k), where n is the number of examples
        and k is the number of classes.
      weights: a `Tensor` of shape (m, k), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      losses for each example.
    """
    pass

  def evaluate_binary_classification(self, predictions, weights):
    """Evaluates a binary classification loss on the given predictions.

    While we do provide a default implementation that thunks down to
    evaluate_multiclass(), subclasses are advised to override this method, since
    it's likely that a hand-written binary classification loss will be more
    efficient that one that works indirectly through a multiclass loss.

    Args:
      predictions: a `Tensor` of shape (n,), where n is the number of examples.
      weights: a `Tensor` of shape (m, 2), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      losses for each example.
    """
    predictions = _convert_to_binary_classification_predictions(predictions)
    return self.evaluate_multiclass(
        tf.stack([0.5 * predictions, -0.5 * predictions], axis=1), weights)


class ZeroOneLoss(MulticlassLoss):
  """Zero-one loss.

  The zero-one loss is normalized and non-differentiable (it's piecewise
  constant).
  """

  @property
  def is_differentiable(self):
    """Returns False, since the zero-one loss is discontinuous."""
    return False

  @property
  def is_normalized(self):
    """Returns True, since the zero-one loss is normalized."""
    return True

  def __hash__(self):
    return hash(type(self))

  def __eq__(self, other):
    return type(other) is type(self)

  def evaluate_multiclass(self, predictions, weights):
    """Evaluates the multiclass zero-one loss on the given predictions.

    Given a rank-2 `Tensor` of predictions with shape (n, k), where n is the
    number of examples and k is the number of classes, and another rank-2
    `Tensor` of weights with shape (m, k), where m is broadcastable to n, this
    method will return a `Tensor` of shape (n,) where the ith element is:

    ```python
    maximum_prediction[i] = max(predictions[i, :])
    maximum_weights[i] = [
      weights[i, j] for j in range(k)
      if predictions[i, j] >= maximum_prediction[i]]
    zero_one_loss[i] = sum(maximum_weights[i]) / len(maximum_weights[i])
    ```

    Args:
      predictions: a `Tensor` of shape (n, k), where n is the number of examples
        and k is the number of classes.
      weights: a `Tensor` of shape (m, k), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      zero-one losses for each example.

    Raises:
      TypeError: if "predictions" is not a floating-point `Tensor`, or "weights"
        is not a `Tensor`.
      ValueError: if "predictions" and "weights" have different numbers of
        columns (i.e. if the number of classes is inconsistent).
    """
    num_classes = helpers.get_num_columns_of_2d_tensor(
        predictions, name="multiclass predictions")
    weights_num_classes = helpers.get_num_columns_of_2d_tensor(
        weights, name="weights")
    if weights_num_classes != num_classes:
      raise ValueError("weights must have the same number of columns as "
                       "predictions ({} vs. {}): did you specify num_classes "
                       "correctly when you created your context?".format(
                           weights_num_classes, num_classes))
    dtype = predictions.dtype.base_dtype
    if not dtype.is_floating:
      raise TypeError("multiclass predictions must be floating-point")

    thresholded_predictions = tf.cast(
        predictions >= tf.reduce_max(predictions, axis=1, keepdims=True),
        dtype=dtype)
    thresholded_predictions /= tf.reduce_sum(
        thresholded_predictions, axis=1, keepdims=True)
    return tf.reduce_sum(
        tf.cast(weights, dtype=dtype) * thresholded_predictions, axis=1)

  def evaluate_binary_classification(self, predictions, weights):
    """Evaluates the zero-one loss on the given predictions.

    Given a rank-1 `Tensor` of predictions with shape (n,), where n is the
    number of examples, and a rank-2 `Tensor` of weights with shape (m, 2),
    where m is broadcastable to n, this method will return a `Tensor` of shape
    (n,) where the ith element is:

    ```python
    zero_one_loss[i] = weights[i, 0] * 1{predictions[i] > 0} +
      0.5 * (weights[i, 0] + weights[i, 1]) * 1{predictions[i] == 0} +
      weights[i, 1] * 1{predictions[i] < 0}
    ```

    where 1{} is an indicator function.

    You can think of weights[:, 0] as being the per-example costs associated
    with making a positive prediction, and weights[:, 1] as those for a negative
    prediction.

    Args:
      predictions: a `Tensor` of shape (n,), where n is the number of examples.
      weights: a `Tensor` of shape (m, 2), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      zero-one losses for each example.

    Raises:
      TypeError: if "predictions" is not a floating-point `Tensor`, or "weights"
        is not a `Tensor`.
      ValueError: if "predictions" is not rank-1, or "weights" is not a rank-2
        `Tensor` with exactly two columns.
    """
    predictions = _convert_to_binary_classification_predictions(predictions)
    columns = helpers.get_num_columns_of_2d_tensor(weights, name="weights")
    if columns != 2:
      raise ValueError("weights must have two columns")
    dtype = predictions.dtype.base_dtype

    positive_weights = tf.cast(weights[:, 0], dtype=dtype)
    negative_weights = tf.cast(weights[:, 1], dtype=dtype)

    sign = tf.sign(predictions)
    return 0.5 * ((positive_weights + negative_weights) + sign *
                  (positive_weights - negative_weights))


class HingeLoss(MulticlassLoss):
  """Hinge loss.

  The hinge loss is subdifferentiable and non-normalized.
  """

  def __init__(self, margin=1.0):
    """Creates a new HingeLoss object with the given margin.

    The margin determines how far a prediction must be from the decision
    boundary in order for it to be penalized. When the margin is zero, this
    threshold is exactly the decision boundary. When the margin is at least one,
    the hinge loss upper bounds the zero-one loss.

    Args:
      margin: non-negative float, the margin of the hinge loss. Defaults to 1.

    Raises:
      ValueError: if the margin is negative.
    """
    super(HingeLoss, self).__init__()

    self._margin = float(margin)
    if margin < 0.0:
      raise ValueError("margin must be non-negative")

  @property
  def margin(self):
    """Accessor for the margin constructor parameter."""
    return self._margin

  @property
  def is_differentiable(self):
    """Returns True, since the hinge loss is subdifferentiable."""
    return True

  @property
  def is_normalized(self):
    """Returns False, since the hinge loss is unbounded."""
    return False

  def __hash__(self):
    return hash((type(self), self._margin))

  def __eq__(self, other):
    return (type(other) is type(self)) and (self._margin == other.margin)

  def evaluate_multiclass(self, predictions, weights):
    """Evaluates the multiclass hinge loss on the given predictions.

    Given a rank-1 `Tensor` of predictions with shape (n,), where n is the
    number of examples, and a rank-2 `Tensor` of weights with shape (m, 2),
    where m is broadcastable to n, this method will return a `Tensor` of shape
    (n,) where the ith element is:

    ```python
    hinge_loss[i] = weights[i, 0] + sum_{j=0}^{num_classes - 2} (
        (weights[i, j+1] - weights[i, j]) * max_{l=j+1}^{num_classes-1}
        max{0, margin + predictions[i, l] - mean_{k=0}^j predictions[i, k]}
    )
    ```

    where we've assumed (without loss of generality) that the weights and
    predictions are ordered in such a way that weights[i, j] <= weights[i, j+1].
    In the implementation, of course, we cannot simply assume this, and actually
    perform a sort.

    This is admittedly a somewhat strange-seeming formulation, and it's
    complicated and expensive to implement. The reason it was chosen is that it
    satisfies the following properties:

    1. It's shift invariant: adding a constant to every weight will shift the
       loss by the same constant.
    2. It's scale invariant: multiplying every weight by a constant will scale
       the loss by the same constant.
    3. When there are only two classes, it's equivalent to the binary hinge loss
       implemented in evaluate_binary_classification().
    4. When the weights represent a misclassification rate (i.e. weights[i, 0] =
       0 and weights[i, j] = 1 for i > 0, assuming the weights are sorted), it's
       equivalent to the usual multiclass hinge misclassification loss.
    5. It's convex in the predictions, and upper bounds the multiclass 0-1 loss
       when margin >= 1.

    Args:
      predictions: a `Tensor` of shape (n, k), where n is the number of examples
        and k is the number of classes.
      weights: a `Tensor` of shape (m, k), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      hinge losses for each example.

    Raises:
      TypeError: if "predictions" is not a floating-point `Tensor`, or "weights"
        is not a `Tensor`.
      ValueError: if "predictions" and "weights" have different numbers of
        columns (i.e. if the number of classes is inconsistent).
    """
    num_classes = helpers.get_num_columns_of_2d_tensor(
        predictions, name="multiclass predictions")
    weights_num_classes = helpers.get_num_columns_of_2d_tensor(
        weights, name="weights")
    if weights_num_classes != num_classes:
      raise ValueError("weights must have the same number of columns as "
                       "predictions ({} vs. {}): did you specify num_classes "
                       "correctly when you created your context?".format(
                           weights_num_classes, num_classes))
    dtype = predictions.dtype.base_dtype
    if not dtype.is_floating:
      raise TypeError("multiclass predictions must be floating-point")
    zero = tf.zeros(1, dtype=dtype)

    weights_rows = tf.shape(weights)[0]
    predictions_rows = tf.shape(predictions)[0]

    # We start out by finding a permutation for each row that will cause the
    # weights to be nondecreasing.
    weights_permutation = tf.argsort(weights, axis=1)
    # This won't work if predictions_rows isn't divisible by weights_rows
    # (tf.stack() below will fail), but we require weights to be broadcastable
    # to predictions (usually, weights_rows will either be 1, or equal to
    # predictions_rows).
    predictions_permutation = tf.tile(weights_permutation,
                                      [predictions_rows / weights_rows, 1])

    # First we create a Tensor of shape [weights_rows, num_classes, 2], for
    # which:
    #   weights_indices[i, j, 0] = i
    #   weights_indices[i, j, 1] = weights_permutation[j]
    # Next, we use gather_nd to re-organize the weights such that:
    #   new_weights[i, j] = old_weights[i, weights_permutation[j]]
    weights_iota = tf.range(weights_rows)
    weights_iota = tf.expand_dims(weights_iota, axis=-1)
    weights_iota = tf.tile(weights_iota, [1, num_classes])
    weights_indices = tf.stack([weights_iota, weights_permutation], axis=2)
    weights = tf.gather_nd(tf.cast(weights, dtype=dtype), weights_indices)

    # Next we create a Tensor of shape [predictions_rows, num_classes, 2], for
    # which:
    #   predictions_indices[i, j, 0] = i
    #   predictions_indices[i, j, 1] = predictions_permutation[j]
    # Next, we use gather_nd to re-organize the predictions such that:
    #   new_predictions[i, j] = old_predictions[i, predictions_permutation[j]]
    predictions_iota = tf.range(predictions_rows)
    predictions_iota = tf.expand_dims(predictions_iota, axis=-1)
    predictions_iota = tf.tile(predictions_iota, [1, num_classes])
    predictions_indices = tf.stack([predictions_iota, predictions_permutation],
                                   axis=2)
    predictions = tf.gather_nd(predictions, predictions_indices)

    # At this point, every row of weights and predictions has been sorted in
    # such a way that the weights are nondecreasing. We wish to calculate the
    # following:
    #   result[i] = weights[i, 0] + \sum_{j=0}^{num_classes - 2} (
    #     (weights[i, j+1] - weights[i, j]) * max_{l=j+1}^{num_classes-1}
    #     max{0, margin + predictions[i, l] - mean_{k=0}^j predictions[i, k]}
    #   )
    # Notice that the innermost max is a hinge.
    result = weights[:, 0]
    for ii in xrange(num_classes - 1):
      scale = weights[:, ii + 1] - weights[:, ii]
      # The "included" predictions are those in the above max over l, and the
      # "excluded" predictions are those in the above mean over k.
      included = predictions[:, (ii + 1):num_classes]
      included = tf.reduce_max(included, axis=1)
      excluded = predictions[:, 0:(ii + 1)]
      excluded = tf.reduce_mean(excluded, axis=1)
      result += scale * tf.maximum(zero, self._margin + included - excluded)

    return result

  def evaluate_binary_classification(self, predictions, weights):
    """Evaluates the hinge loss on the given predictions.

    Given a rank-1 `Tensor` of predictions with shape (n,), where n is the
    number of examples, and a rank-2 `Tensor` of weights with shape (m, 2),
    where m is broadcastable to n, this method will return a `Tensor` of shape
    (n,) where the ith element is:

    ```python
    hinge_loss[i] = constant_weights[i] +
      (weights[i, 0] - constant_weights[i]) * max{0, margin + predictions[i]} +
      (weights[i, 1] - constant_weights[i]) * max{0, margin - predictions[i]}
    ```

    where constant_weights[i] = min{weights[i, 0], weights[i, 1]} contains the
    minimum weights.

    You can think of weights[:, 0] as being the per-example costs associated
    with making a positive prediction, and weights[:, 1] as those for a negative
    prediction.

    Args:
      predictions: a `Tensor` of shape (n,), where n is the number of examples.
      weights: a `Tensor` of shape (m, 2), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      hinge losses for each example.

    Raises:
      TypeError: if "predictions" is not a floating-point `Tensor`, or "weights"
        is not a `Tensor`.
      ValueError: if "predictions" is not rank-1, or "weights" is not a rank-2
        `Tensor` with exactly two columns.
    """
    predictions = _convert_to_binary_classification_predictions(predictions)
    columns = helpers.get_num_columns_of_2d_tensor(weights, name="weights")
    if columns != 2:
      raise ValueError("weights must have two columns")
    dtype = predictions.dtype.base_dtype
    zero = tf.zeros(1, dtype=dtype)

    positive_weights = tf.cast(weights[:, 0], dtype=dtype)
    negative_weights = tf.cast(weights[:, 1], dtype=dtype)
    constant_weights = tf.minimum(positive_weights, negative_weights)
    positive_weights -= constant_weights
    negative_weights -= constant_weights

    is_positive = tf.maximum(zero, self._margin + predictions)
    is_negative = tf.maximum(zero, self._margin - predictions)

    return constant_weights + (
        positive_weights * is_positive + negative_weights * is_negative)


class SoftmaxLoss(MulticlassLoss):
  """Softmax loss.

  The softmax loss is subdifferentiable and normalized.
  """

  @property
  def is_differentiable(self):
    """Returns True, since the softmax loss is differentiable."""
    return True

  @property
  def is_normalized(self):
    """Returns True, since the softmax loss is an expected zero-one loss."""
    return True

  def __hash__(self):
    return hash(type(self))

  def __eq__(self, other):
    return type(other) is type(self)

  def evaluate_multiclass(self, predictions, weights):
    """Evaluates the multiclass softmax loss on the given predictions.

    Given a rank-1 `Tensor` of predictions with shape (n,), where n is the
    number of examples, and a rank-2 `Tensor` of weights with shape (m, 2),
    where m is broadcastable to n, this method will return a `Tensor` of shape
    (n,) where the ith element is:

    ```python
    softmax_loss[i] = sum_j ( weights[i, j] * (
        exp(predictions[i, j]) / sum_k exp(predictions[i, k]) ) )
    ```

    Args:
      predictions: a `Tensor` of shape (n, k), where n is the number of examples
        and k is the number of classes.
      weights: a `Tensor` of shape (m, k), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      softmax losses for each example.

    Raises:
      TypeError: if "predictions" is not a floating-point `Tensor`, or "weights"
        is not a `Tensor`.
      ValueError: if "predictions" and "weights" have different numbers of
        columns (i.e. if the number of classes is inconsistent).
    """
    num_classes = helpers.get_num_columns_of_2d_tensor(
        predictions, name="multiclass predictions")
    weights_num_classes = helpers.get_num_columns_of_2d_tensor(
        weights, name="weights")
    if weights_num_classes != num_classes:
      raise ValueError("weights must have the same number of columns as "
                       "predictions ({} vs. {}): did you specify num_classes "
                       "correctly when you created your context?".format(
                           weights_num_classes, num_classes))
    dtype = predictions.dtype.base_dtype
    if not dtype.is_floating:
      raise TypeError("multiclass predictions must be floating-point")

    maximum_predictions = tf.reduce_max(predictions, axis=1, keepdims=True)
    numerators = tf.exp(predictions - maximum_predictions)
    denominators = tf.reduce_sum(numerators, axis=1, keepdims=True)
    probabilities = numerators / denominators

    weights = tf.cast(weights, dtype=dtype)
    return tf.reduce_sum(weights * probabilities, axis=1)

  def evaluate_binary_classification(self, predictions, weights):
    """Evaluates the softmax loss on the given predictions.

    Given a rank-1 `Tensor` of predictions with shape (n,), where n is the
    number of examples, and a rank-2 `Tensor` of weights with shape (m, 2),
    where m is broadcastable to n, this method will return a `Tensor` of shape
    (n,) where the ith element is:

    ```python
    softmax_loss[i] = (
      weights[i, 0] * ( exp(predictions[i]) / ( 1 + exp(predictions[i]) ) ) +
      weights[i, 1] * ( 1 / ( 1 + exp(predictions[i]) ) ) )
    ```

    where constant_weights[i] = min{weights[i, 0], weights[i, 1]} contains the
    minimum weights.

    You can think of weights[:, 0] as being the per-example costs associated
    with making a positive prediction, and weights[:, 1] as those for a negative
    prediction.

    Args:
      predictions: a `Tensor` of shape (n,), where n is the number of examples.
      weights: a `Tensor` of shape (m, 2), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      softmax losses for each example.

    Raises:
      TypeError: if "predictions" is not a floating-point `Tensor`, or "weights"
        is not a `Tensor`.
      ValueError: if "predictions" is not rank-1, or "weights" is not a rank-2
        `Tensor` with exactly two columns.
    """
    predictions = _convert_to_binary_classification_predictions(predictions)
    columns = helpers.get_num_columns_of_2d_tensor(weights, name="weights")
    if columns != 2:
      raise ValueError("weights must have two columns")
    dtype = predictions.dtype.base_dtype

    positive_weights = tf.cast(weights[:, 0], dtype=dtype)
    negative_weights = tf.cast(weights[:, 1], dtype=dtype)

    is_positive = tf.sigmoid(predictions)
    is_negative = tf.sigmoid(-predictions)

    return positive_weights * is_positive + negative_weights * is_negative


class SoftmaxCrossEntropyLoss(MulticlassLoss):
  """Softmax cross-entropy loss.

  The softmax cross-entropy loss is subdifferentiable and non-normalized.
  """

  @property
  def is_differentiable(self):
    """Returns True, since the softmax cross-entropy loss is differentiable."""
    return True

  @property
  def is_normalized(self):
    """Returns False, since the softmax cross-entropy loss is unbounded."""
    return False

  def __hash__(self):
    return hash(type(self))

  def __eq__(self, other):
    return type(other) is type(self)

  def evaluate_multiclass(self, predictions, weights):
    """Evaluates the multiclass cross-entropy loss on the given predictions.

    Given a rank-1 `Tensor` of predictions with shape (n,), where n is the
    number of examples, and a rank-2 `Tensor` of weights with shape (m, 2),
    where m is broadcastable to n, this method will return a `Tensor` of shape
    (n,) where the ith element is:

    ```python
    max_weight[i] = max_j weights[i, j]
    softmax_cross_entropy_loss[i] = max_weight[i] - sum_j (
      (max_weight[i] - weights[i, j]) * ( 1 +
        log( exp(predictions[i, j]) / sum_k exp(predictions[i, k]) ) ) )
    ```

    The reason this formulation was chosen is that it can be derived from the
    softmax loss using the inequality -log(p) >= 1-p. Indeed, the difference
    between the two losses is entirely due to the slop in this inequality. In
    particular, it satisfies the following properties:

    1. It's shift invariant: adding a constant to every weight will shift the
       loss by the same constant.
    2. It's scale invariant: multiplying every weight by a constant will scale
       the loss by the same constant.
    3. When there are only two classes, it's equivalent to the binary softmax
       cross-entropy loss implemented in evaluate_binary_classification().
    4. When the weights represent an expected misclassification rate (i.e.
       weights[i, j] >= 0, and sum_j weights[i, j] = 1), it's equivalent to the
       usual multiclass softmax cross-entropy misclassification loss.
    5. It's convex in the predictions, and upper bounds the multiclass softmax
       loss.

    Args:
      predictions: a `Tensor` of shape (n, k), where n is the number of examples
        and k is the number of classes.
      weights: a `Tensor` of shape (m, k), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      softmax cross-entropy losses for each example.

    Raises:
      TypeError: if "predictions" is not a floating-point `Tensor`, or "weights"
        is not a `Tensor`.
      ValueError: if "predictions" and "weights" have different numbers of
        columns (i.e. if the number of classes is inconsistent).
    """
    num_classes = helpers.get_num_columns_of_2d_tensor(
        predictions, name="multiclass predictions")
    weights_num_classes = helpers.get_num_columns_of_2d_tensor(
        weights, name="weights")
    if weights_num_classes != num_classes:
      raise ValueError("weights must have the same number of columns as "
                       "predictions ({} vs. {}): did you specify num_classes "
                       "correctly when you created your context?".format(
                           weights_num_classes, num_classes))
    dtype = predictions.dtype.base_dtype
    if not dtype.is_floating:
      raise TypeError("multiclass predictions must be floating-point")

    maximum_predictions = tf.reduce_max(predictions, axis=1, keepdims=True)
    numerators = tf.exp(predictions - maximum_predictions)
    denominators = tf.reduce_sum(numerators, axis=1, keepdims=True)
    log_probabilities = tf.math.log(numerators / denominators)

    weights = tf.cast(weights, dtype=dtype)
    maximum_weights = tf.reduce_max(weights, axis=1, keepdims=True)
    return (tf.squeeze(maximum_weights) - tf.reduce_sum(
        (maximum_weights - weights) * (1 + log_probabilities), axis=1))

  def evaluate_binary_classification(self, predictions, weights):
    """Evaluates the cross-entropy loss on the given predictions.

    Given a rank-1 `Tensor` of predictions with shape (n,), where n is the
    number of examples, and a rank-2 `Tensor` of weights with shape (m, 2),
    where m is broadcastable to n, this method will return a `Tensor` of shape
    (n,) where the ith element is:

    ```python
    softmax_cross_entropy_loss[i] = ( constant_weights[i] +
      (weights[i, 0] - constant_weights[i]) * log(1 + exp(predictions[i]) +
      (weights[i, 1] - constant_weights[i]) * log(1 + exp(-predictions[i]) )
    ```

    where constant_weights[i] = min{weights[i, 0], weights[i, 1]} contains the
    minimum weights.

    You can think of weights[:, 0] as being the per-example costs associated
    with making a positive prediction, and weights[:, 1] as those for a negative
    prediction.

    Args:
      predictions: a `Tensor` of shape (n,), where n is the number of examples.
      weights: a `Tensor` of shape (m, 2), where m is broadcastable to n. This
        `Tensor` is *not* necessarily non-negative.

    Returns:
      A `Tensor` of shape (n,) and dtype=predictions.dtype, containing the
      softmax cross-entropy losses for each example.

    Raises:
      TypeError: if "predictions" is not a floating-point `Tensor`, or "weights"
        is not a `Tensor`.
      ValueError: if "predictions" is not rank-1, or "weights" is not a rank-2
        `Tensor` with exactly two columns.
    """
    predictions = _convert_to_binary_classification_predictions(predictions)
    columns = helpers.get_num_columns_of_2d_tensor(weights, name="weights")
    if columns != 2:
      raise ValueError("weights must have two columns")
    dtype = predictions.dtype.base_dtype
    zeros = tf.zeros_like(predictions)

    positive_weights = tf.cast(weights[:, 0], dtype=dtype)
    negative_weights = tf.cast(weights[:, 1], dtype=dtype)
    constant_weights = tf.minimum(positive_weights, negative_weights)
    positive_weights -= constant_weights
    negative_weights -= constant_weights

    # We use tf.where() instead of tf.abs() and tf.maximum() since, if we
    # didn't, then we would have zero gradients whenever predictions=0, with the
    # consequence that optimization could get "stuck".
    condition = (predictions <= zeros)
    absolute_predictions = tf.where(condition, -predictions, predictions)
    intermediate = tf.math.log(1 + tf.exp(-absolute_predictions))

    # Notice that:
    #   is_positive = log(1 + exp(-|predictions|)) + max{0, predictions}
    #   is_positive =
    #     log(1 + exp(predictions))                   if (predictions <= 0)
    #     log(1 + exp(-predictions)) + predictions    if (predictions >= 0)
    #   is_positive = log(1 + exp(predictions))
    # Likewise:
    #   is_negative = log(1 + exp(-predictions))
    # The reason for representing these in terms of "intermediate" is to improve
    # numerical accuracy.
    is_positive = intermediate + tf.where(condition, zeros, predictions)
    is_negative = intermediate + tf.where(condition, -predictions, zeros)

    return constant_weights + (
        positive_weights * is_positive + negative_weights * is_negative)
