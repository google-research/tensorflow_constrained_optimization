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
"""Contains the `Predicate` class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import helpers


class Predicate(helpers.RateObject):
  """Object representing a boolean predicate.

  We actually support *soft* predicates: the wrapped `Tensor` is allowed to take
  on values between zero and one, not just the two extremes.
  """

  def __init__(self, tensor):
    """Creates a new `Predicate`.

    Args:
      tensor: an object convertible to a rank-1 `Tensor` (e.g. a scalar, list,
        numpy array, or a `Tensor` itself). All elements will be clipped to the
        range [0,1].
    """
    self._tensor = tf.cast(
        helpers.convert_to_1d_tensor(tensor, name="predicate"),
        dtype=tf.float32)

  @property
  def tensor(self):
    """Returns the predicate as a `Tensor`.

    Returns:
      A rank one `Tensor` with dtype tf.float32. All of its values will be in
      the range [0,1].
    """
    # We perform the clipping here, instead of in __init__, so that we don't
    # clip Tensors that have already been clipped. This doesn't affect the
    # result, regardless of what operations have been performed on the
    # Predicate.
    return tf.clip_by_value(self._tensor, 0.0, 1.0)

  def __invert__(self):
    """Returns the logical NOT of this `Predicate`.

    Since `Predicate` objects support *soft* predicates represented as
    floating-point `Tensor`s between zero and one, we actually return 1 -
    predicate.

    Returns:
      A `Predicate` representing the logical NOT of this `Predicate`.
    """
    return Predicate(1.0 - self._tensor)

  def __and__(self, other):
    """Returns the logical AND of two `Predicate`s.

    Since `Predicate` objects support *soft* predicates represented as
    floating-point `Tensor`s between zero and one, we actually return
    min(predicate1, predicate2).

    Args:
      other: `Predicate` object to AND with this object.

    Returns:
      A `Predicate` representing the logical AND of the two `Predicate`s.

    Raises:
      TypeError: if the "other" parameter is not a `Predicate` object.
    """
    if not isinstance(other, Predicate):
      raise TypeError("Predicates can only be ANDed with other Predicates")
    # We use min(a,b) instead of e.g. a*b because we want to preserve the usual
    # logical identities. We pass the private _tensors to tf.minimum, instead
    # of using the public predicate property, so that we don't clip quantities
    # that have already been clipped.
    #
    # pylint: disable=protected-access
    return Predicate(tf.minimum(self._tensor, other._tensor))
    # pylint: enable=protected-access

  def __or__(self, other):
    """Returns the logical OR of two `Predicate`s.

    Since `Predicate` objects support *soft* predicates represented as
    floating-point `Tensor`s between zero and one, we actually return
    max(predicate1, predicate2).

    Args:
      other: `Predicate` object to OR with this object.

    Returns:
      A `Predicate` representing the logical OR of the two `Predicate`s.

    Raises:
      TypeError: if the "other" parameter is not a `Predicate` object.
    """
    if not isinstance(other, Predicate):
      raise TypeError("Predicates can only be ORed with other Predicates")
    # We use max(a,b) instead of e.g. 1-(1-a)*(1-b) because we want to preserve
    # the usual logical identities. We pass the private _tensors to tf.minimum,
    # instead of using the public predicate property, so that we don't clip
    # quantities that have already been clipped.
    #
    # pylint: disable=protected-access
    return Predicate(tf.maximum(self._tensor, other._tensor))
    # pylint: enable=protected-access

  def __xor__(self, other):
    """Returns the logical XOR of two `Predicate`s.

    Since `Predicate` objects support *soft* predicates represented as
    floating-point `Tensor`s between zero and one, we actually return
    max(min(predicate1, 1-predicate2), min(1-predicate1, predicate2)).

    Args:
      other: `Predicate` object to XOR with this object.

    Returns:
      A `Predicate` representing the logical XOR of the two `Predicate`s.

    Raises:
      TypeError: if the "other" parameter is not a `Predicate` object.
    """
    if not isinstance(other, Predicate):
      raise TypeError("Predicates can only be XORed with other Predicates")
    return (self & ~other) | (~self & other)
