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
"""Contains the `Constraint` and `ConstraintList` classes.

A `Constraint` is nothing but a thin wrapper around an `Expression`, and
represents the constraint that the wrapped `Expression` should be non-positive.

A `ConstraintList` is basically a stripped-down list of `Constraint` objects,
with the additional property that if one attempts to add a duplicate
`Constraint` to the list, then it will be ignored. In the past, we used a set,
but realized that it's desirable for the constraints to appear in the resulting
`RateMinimizationProblem` in the order in which they were added.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_constrained_optimization.python.rates import helpers


class Constraint(helpers.RateObject):
  """Represents an inequality constraint.

  This class is nothing but a thin wrapper around an `Expression`, and
  represents the constraint that the wrapped expression is non-positive.
  """

  def __init__(self, expression):
    """Constructs a new `Constraint` object from an `Expression`."""
    self._expression = expression

  @property
  def expression(self):
    """Returns the `Expression` that is constrained to be non-positive."""
    return self._expression


class ConstraintList(helpers.RateObject):
  """Represents a list of `Constraint`s.

  Aside from having a very stripped-down interface compared to a normal Python
  list, this class also differs in that (i) it verifies that every element it
  contains is a `Constraint` object, and (ii) if one tries to add a duplicate
  `Constraint` to the list, then it will be ignored.
  """

  def __init__(self, container=None):
    """Creates a new `ConstraintList` from a collection of `Constraint`s."""
    self._list = []
    if container is not None:
      for element in container:
        self.append(element)

  @property
  def list(self):
    """Returns the list of `Constraint`s as a Python list."""
    # We take a slice of the whole list to make a copy.
    return self._list[:]

  def append(self, element):
    """Appends a new `Constraint` to the list, ignoring duplicates."""
    if not isinstance(element, Constraint):
      raise TypeError("every element added to a ConstraintList must be a "
                      "Constraint object")
    # FUTURE WORK: this linear-time duplicate check is probably fine, since we
    # expect to generally have very few constraints, but it would be nice to use
    # something that scales better.
    if element not in self._list:
      self._list.append(element)

  def __eq__(self, other):
    return self._list.__eq__(other._list)  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self.__eq__(other)

  def __len__(self):
    """Returns the length of this `ConstraintList`."""
    return self._list.__len__()

  def __iter__(self):
    """Returns an iterator over the wrapped list of `Constraint`s."""
    return self._list.__iter__()

  def __add__(self, other):
    """Appends two `ConstraintList`s."""
    result = ConstraintList(self)
    for element in other:
      result.append(element)
    return result

  def __radd__(self, other):
    """Appends two `ConstraintList`s."""
    result = ConstraintList(other)
    for element in self:
      result.append(element)
    return result

  def __getitem__(self, slice_spec):
    """Returns a single element or a slice of this `ConstraintList`."""
    result = self._list[slice_spec]
    if isinstance(result, list):
      result = ConstraintList(result)
    return result
