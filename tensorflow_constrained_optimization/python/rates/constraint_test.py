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
"""Tests for constraint.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import constraint


class ConstraintListTest(tf.test.TestCase):
  """Tests for `ConstraintList` classes."""

  def test_construct(self):
    """Tests the `ConstraintList` constructor."""
    # We construct all constraints with `None` expressions, since we'll only be
    # comparing them by object ID, not value.
    constraint1 = constraint.Constraint(None)
    constraint2 = constraint.Constraint(None)
    constraint3 = constraint1
    constraint4 = constraint.Constraint(None)
    constraint5 = constraint4
    constraint6 = constraint.Constraint(None)

    constraint_list = constraint.ConstraintList([
        constraint1, constraint2, constraint3, constraint4, constraint5,
        constraint6
    ])
    self.assertEqual(4, len(constraint_list))
    self.assertEqual([constraint1, constraint2, constraint4, constraint6],
                     constraint_list.list)

  def test_append_raises(self):
    """Tests that "append" raises when given a non-Constraint."""
    constraint_list = constraint.ConstraintList()
    self.assertEqual(0, len(constraint_list))
    self.assertEqual([], constraint_list.list)

    with self.assertRaises(TypeError):
      constraint_list.append(42)

  def test_add(self):
    """Tests `ConstraintList`'s "__add__" method."""
    # We construct all constraints with `None` expressions, since we'll only be
    # comparing them by object ID, not value.
    constraint1 = constraint.Constraint(None)
    constraint2 = constraint.Constraint(None)
    constraint3 = constraint1
    constraint4 = constraint.Constraint(None)
    constraint5 = constraint4
    constraint6 = constraint.Constraint(None)

    lhs = [constraint1, constraint2]
    rhs = [constraint3, constraint4, constraint5, constraint6]

    constraint_list = constraint.ConstraintList(lhs)
    self.assertEqual(2, len(constraint_list))
    self.assertEqual([constraint1, constraint2], constraint_list.list)

    constraint_list += rhs
    self.assertEqual(4, len(constraint_list))
    self.assertEqual([constraint1, constraint2, constraint4, constraint6],
                     constraint_list.list)

  def test_radd(self):
    """Tests `ConstraintList`'s "__radd__" method."""
    # We construct all constraints with `None` expressions, since we'll only be
    # comparing them by object ID, not value.
    constraint1 = constraint.Constraint(None)
    constraint2 = constraint.Constraint(None)
    constraint3 = constraint1
    constraint4 = constraint.Constraint(None)
    constraint5 = constraint4
    constraint6 = constraint.Constraint(None)

    lhs = [constraint1, constraint2]
    rhs = [constraint3, constraint4, constraint5, constraint6]

    constraint_list = constraint.ConstraintList(rhs)
    self.assertEqual(3, len(constraint_list))
    self.assertEqual([constraint1, constraint4, constraint6],
                     constraint_list.list)

    constraint_list = lhs + constraint_list
    self.assertEqual(4, len(constraint_list))
    self.assertEqual([constraint1, constraint2, constraint4, constraint6],
                     constraint_list.list)


if __name__ == "__main__":
  tf.test.main()
