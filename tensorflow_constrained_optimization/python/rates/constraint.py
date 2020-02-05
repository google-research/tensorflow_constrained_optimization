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


class ConstraintList(helpers.UniqueList):
  """Represents a list of `Constraint`s.

  Aside from having a very stripped-down interface compared to a normal Python
  list, this class also differs in that (i) it verifies that every element it
  contains is a `Constraint` object, and (ii) duplicate elements are removed
  (but, unlike a set, order is preserved).
  """

  def __init__(self, collection=None):
    """Creates a new `ConstraintList` from a collection."""
    super(ConstraintList, self).__init__(
        collection=collection, element_type=Constraint)
