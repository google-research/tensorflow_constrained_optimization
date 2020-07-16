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

import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import helpers


class Predicate(helpers.RateObject):
  """Object representing a boolean predicate.

  We actually support *soft* predicates: the wrapped `Tensor` is allowed to take
  on values between zero and one, not just the two extremes.
  """

  @classmethod
  def _convert_and_clip_fn(cls, arg):
    """Converts the given object to a rank-one float32 `Tensor` in [0,1]."""
    return tf.clip_by_value(
        tf.cast(
            helpers.convert_to_1d_tensor(arg, "predicate"), dtype=tf.float32),
        0.0, 1.0)

  @classmethod
  def _invert_fn(cls, tensor):
    """Returns 1 - tensor."""
    # We have this in a class method, instead of a lambda, so that there will
    # only be one copy of the function, causing derived DeferredTensors found by
    # inverting the same original DeferredTensor to be considered equal.
    return 1.0 - tensor

  def __init__(  # pylint: disable=invalid-name
      self, tensor, _convert_and_clip=True):
    """Creates a new `Predicate`.

    Args:
      tensor: an object convertible to a rank-1 `Tensor` (e.g. a scalar, list,
        numpy array, or a `Tensor` itself), or a nullary function returning such
        an object, or a DeferredTensor. This object will be converted to a
        float32 `DeferredTensor` and clipped to [0,1].
      _convert_and_clip: private Boolean. If False, "tensor" will not be
        converted to a float32 `DeferredTensor` and clipped. This is for
        internal use *only*.
    """
    if isinstance(tensor, Predicate):
      raise ValueError("cannot create a Predicate from a Predicate")

    self._tensor = tensor
    if not isinstance(self._tensor, deferred_tensor.DeferredTensor):
      self._tensor = deferred_tensor.ExplicitDeferredTensor(self._tensor)
    if _convert_and_clip:
      self._tensor = deferred_tensor.DeferredTensor.apply(
          Predicate._convert_and_clip_fn, self._tensor)

  @property
  def tensor(self):
    """Returns the predicate as a `DeferredTensor`.

    Returns:
      A rank one `DeferredTensor` with dtype tf.float32. All of its values will
      be in the range [0,1].
    """
    return self._tensor

  def __hash__(self):
    return hash(self._tensor)

  def __eq__(self, other):
    if not isinstance(other, Predicate):
      return False
    return self._tensor == other.tensor

  def __ne__(self, other):
    return not self.__eq__(other)

  def __invert__(self):
    """Returns the logical NOT of this `Predicate`.

    Since `Predicate` objects support *soft* predicates represented as
    floating-point `Tensor`s between zero and one, we actually return 1 -
    predicate.

    Returns:
      A `Predicate` representing the logical NOT of this `Predicate`.
    """
    # We pass "_convert_and_clip=False" since the argument we pass to the
    # Predicate constructor is already a 1d float32 DeferredTensor in [0,1].
    return Predicate(
        deferred_tensor.DeferredTensor.apply(Predicate._invert_fn,
                                             self._tensor),
        _convert_and_clip=False)

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
    # logical identities. We pass "_convert_and_clip=False" since the argument
    # we pass to the Predicate constructor is already a 1d float32
    # DeferredTensor in [0,1].
    return Predicate(
        deferred_tensor.DeferredTensor.apply(tf.minimum, self._tensor,
                                             other.tensor),
        _convert_and_clip=False)

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
    # the usual logical identities. We pass "_convert_and_clip=False" since the
    # argument we pass to the Predicate constructor is already a 1d float32
    # DeferredTensor in [0,1].
    return Predicate(
        deferred_tensor.DeferredTensor.apply(tf.maximum, self._tensor,
                                             other.tensor),
        _convert_and_clip=False)

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
