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
    raise ValueError("%s must have a known rank" % name)

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
  if not tf.is_tensor(tensor):
    raise TypeError("%s must be a Tensor" % name)

  dims = tensor.shape.dims
  if dims is None:
    raise ValueError("%s must have a known rank" % name)
  if len(dims) != 2:
    raise ValueError("%s must be rank 2 (it is rank %d)" % (name, len(dims)))

  columns = dims[1].value
  if columns is None:
    raise ValueError("%s must have a known number of columns" % name)

  return columns


class UniqueList(RateObject):
  """Represents a list of unique elements.

  Aside from having a very stripped-down interface compared to a normal Python
  list, this class also differs in that (i) if the "element_type" constructor
  parameter is provided, it verifies that every element it contains is of the
  given type, and (ii) duplicate elements are removed (but, unlike a set, order
  is preserved).
  """

  def __init__(self, collection=None, element_type=None):
    """Creates a new `UniqueList` from a collection."""
    self._element_type = element_type
    self._set = set()
    self._list = []
    if collection is not None:
      for element in collection:
        self.append(element)

  @property
  def list(self):
    """Returns the contents as a Python list."""
    # We take a slice of the whole list to make a copy.
    return self._list[:]

  def append(self, element):
    """Appends a new element to the list, ignoring duplicates."""
    if (self._element_type is not None and
        not isinstance(element, self._element_type)):
      raise TypeError("every element added to a UniqueList must be of the "
                      "given element_type")
    if element not in self._set:
      self._set.add(element)
      self._list.append(element)

  def __eq__(self, other):
    # Two UniqueLists are equal iff they have the same element_type, and contain
    # the same elements in the same order.
    if self._element_type != other.element_type:
      return False
    return self._list.__eq__(other._list)  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self.__eq__(other)

  def __len__(self):
    """Returns the length of this `UniqueList`."""
    result = self._set.__len__()
    assert result == self._list.__len__()
    return result

  def __iter__(self):
    """Returns an iterator over the wrapped list."""
    return self._list.__iter__()

  def __add__(self, other):
    """Appends two `UniqueList`s."""
    result = UniqueList(self)
    for element in other:
      result.append(element)
    return result

  def __radd__(self, other):
    """Appends two `UniqueList`s."""
    result = UniqueList(other)
    for element in self:
      result.append(element)
    return result

  def __getitem__(self, slice_spec):
    """Returns a single element or a slice of this `UniqueList`."""
    result = self._list[slice_spec]
    if isinstance(result, list):
      result = UniqueList(result)
    return result
