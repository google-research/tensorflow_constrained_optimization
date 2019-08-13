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
"""Contains internal helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class RateObject(object):
  """Empty base class for all rate objects.

  Every class involving the rate helpers will inherit from this one. This
  enables us to easily check, using isinstance(..., RateObject), whether an
  object comes from inside or outside of this library. This is especially useful
  for distinguishing between input types (e.g. `Tensor`s, numpy arrays, scalars,
  lists, etc.) and internal abstract representations of operations (e.g. `Term`,
  `BasicExpression`, `Expression`), and thereby ensuring that each class of
  types is encountered only where it is expected.
  """


def tensors_equal(left, right):
  """Returns true if (but not only if) two objects are equal.

  The purpose of this function is to check whether two `Tensor`s are equal. For
  actual `Tensor` objects, we can't check for value-equality without running a
  TensorFlow op, so all that we can do is check if they're the same object.

  However, many of our functions expect `Tensor`s as arguments, but also accept
  other types, such as python scalars, collections, or numpy arrays, that will
  be converted to `Tensor`s. These types *can* be checked for value equality,
  and this function will do so.

  Args:
    left: an object convertible to a `Tensor` (e.g. a scalar, list, numpy array,
      or a `Tensor` itself).
    right: an object convertible to a `Tensor` (e.g. a scalar, list, numpy
      array, or a `Tensor` itself).

  Returns:
    True if the two arguments are equal. If this function returns False, then it
    was unable to establish that the two objects are equal (but they may be
    equal, anyway).
  """
  if (tf.contrib.framework.is_tensor(left) or
      tf.contrib.framework.is_tensor(right)):
    # We can't compare the values of Tensors, so the best we can do is checking
    # if they're the same object.
    return left is right
  else:
    # Every other allowed type can be handled by numpy.
    return np.array_equal(left, right)


def convert_to_1d_tensor(tensor, name=None):
  """Converts the given object to a rank-one `Tensor`.

  This function checks that the object is trivially convertible to rank-1, i.e.
  has only one "nontrivial" dimension (e.g. the shapes [1000] and [1, 1, None,
  1] are allowed, but [None, 1, None] and [50, 10] are not). If it satisfies
  this condition, then it is returned as a rank-1 `Tensor`.

  Args:
    tensor: an object convertible to a `Tensor` (e.g. a scalar, list, numpy
      array, or a `Tensor` itself).
    name: str, how to refer to the tensor in error messages.

  Returns:
    A rank-1 `Tensor`.

  Raises:
    ValueError: if the resulting tensor is not rank-1.
  """
  tensor = tf.convert_to_tensor(tensor, name=name)
  name = name or "tensor"

  dims = tensor.shape.dims
  if dims is None:
    raise ValueError("%s must have a known shape" % name)

  # If the Tensor is already rank-1, then return it without wrapping a
  # tf.reshape() around it.
  if len(dims) == 1:
    return tensor

  if sum(dim.value != 1 for dim in dims) > 1:
    raise ValueError("all but one dimension of %s must have size one" % name)

  # Reshape to a rank-1 Tensor.
  return tf.reshape(tensor, (-1,))


def get_num_columns_of_2d_tensor(tensor, name="tensor"):
  """Gets the number of columns of a rank-two `Tensor`.

  Args:
    tensor: a rank-2 `Tensor` with a known number of columns.
    name: str, how to refer to the tensor in error messages.

  Returns:
    The number of columns in the tensor.

  Raises:
    TypeError: if "tensor" is not a `Tensor`.
    ValueError: if "tensor" is not a rank-2 `Tensor` with a known number of
      columns.
  """
  if not tf.contrib.framework.is_tensor(tensor):
    raise TypeError("%s must be a tensor" % name)

  dims = tensor.shape.dims
  if dims is None:
    raise ValueError("%s must have known shape" % name)
  if len(dims) != 2:
    raise ValueError("%s must be rank 2 (it is rank %d)" % (name, len(dims)))

  columns = dims[1].value
  if columns is None:
    raise ValueError("%s must have a known number of columns" % name)

  return columns


class Predicate(object):
  """Object representing a boolean predicate.

  We actually support *soft* predicates: the wrapped `Tensor` is allowed to take
  on values between zero and one, not just the two extremes.
  """

  def __init__(self, predicate):
    """Creates a new `Predicate`.

    Args:
      predicate: an object convertible to a rank-1 `Tensor` (e.g. a scalar,
        list, numpy array, or a `Tensor` itself). All elements will be clipped
        to the range [0,1].
    """
    self._predicate = tf.cast(
        convert_to_1d_tensor(predicate, name="predicate"), dtype=tf.float32)

  @property
  def predicate(self):
    """Returns the predicate `Tensor`.

    Returns:
      A rank one `Tensor` with dtype tf.float32. All of its values will be in
      the range [0,1].
    """
    # We perform the clipping here, instead of in __init__, so that we don't
    # clip Tensors that have already been clipped. This doesn't affect the
    # result, regardless of what operations have been performed on the
    # Predicate.
    return tf.clip_by_value(self._predicate, 0.0, 1.0)

  def __invert__(self):
    """Returns the logical NOT of this `Predicate`.

    Since `Predicate` objects support *soft* predicates represented as
    floating-point `Tensor`s between zero and one, we actually return 1 -
    predicate.

    Returns:
      A `Predicate` representing the logical NOT of this `Predicate`.
    """
    return Predicate(1.0 - self._predicate)

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
    # logical identities. We pass the private _predicates to tf.minimum, instead
    # of using the public predicate property, so that we don't clip quantities
    # that have already been clipped.
    #
    # pylint: disable=protected-access
    return Predicate(tf.minimum(self._predicate, other._predicate))
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
    # the usual logical identities. We pass the private _predicates to
    # tf.minimum, instead of using the public predicate property, so that we
    # don't clip quantities that have already been clipped.
    #
    # pylint: disable=protected-access
    return Predicate(tf.maximum(self._predicate, other._predicate))
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
