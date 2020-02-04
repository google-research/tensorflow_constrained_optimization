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
"""Tests for helpers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import helpers

# These tests use some placeholder Tensors, so we want to make sure that they
# execute in graph mode.
tf.compat.v1.disable_eager_execution()


class HelpersTest(tf.test.TestCase):
  """Tests for helper functions in helpers.py."""

  def test_convert_to_1d_tensor(self):
    """Tests the "convert_to_1d_tensor" function."""
    self.assertFalse(tf.executing_eagerly())

    # Trying to make a rank-1 Tensor from a 0d Tensor should succeed.
    expected = [2.7]
    actual = helpers.convert_to_1d_tensor(2.7)
    with self.session() as session:
      self.assertAllClose(expected, session.run(actual), rtol=0, atol=1e-6)

    # Trying to make a rank-1 Tensor from a rank-1 Tensor should succeed.
    expected = [-6.3, 1.0, 5.1]
    actual = helpers.convert_to_1d_tensor(expected)
    with self.session() as session:
      self.assertAllClose(expected, session.run(actual), rtol=0, atol=1e-6)

    # Trying to make a rank-1 Tensor from a shape-(1,2,1) Tensor should succeed
    # (only one of the dimensions is nontrivial).
    expected = [3.6, -1.7]
    actual = helpers.convert_to_1d_tensor([[[3.6], [-1.7]]])
    with self.session() as session:
      self.assertAllClose(expected, session.run(actual), rtol=0, atol=1e-6)

    # Trying to make a rank-1 Tensor from a shape-(1,None,1) Tensor should
    # succeed (only one of the dimensions is nontrivial).
    expected = [0.2, -2.4, 0.0]
    placeholder = tf.compat.v1.placeholder(tf.float32, shape=(1, None, 1))
    actual = helpers.convert_to_1d_tensor(placeholder)
    with self.session() as session:
      self.assertAllClose(
          expected,
          session.run(
              actual, feed_dict={placeholder: [[[0.2], [-2.4], [0.0]]]}),
          rtol=0,
          atol=1e-6)

    # Trying to make a rank-1 Tensor from a rank-2 Tensor should fail.
    with self.assertRaises(ValueError):
      _ = helpers.convert_to_1d_tensor([[1, 2], [3, 4]])

    # Trying to make a rank-1 Tensor from a shape-(None,2) Tensor should fail.
    placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
    with self.assertRaises(ValueError):
      _ = helpers.convert_to_1d_tensor(placeholder)

  def test_get_num_columns_of_2d_tensor(self):
    """Tests the "get_num_columns_of_2d_tensor" function."""
    self.assertFalse(tf.executing_eagerly())

    # Trying to get the number of columns from a non-tensor should fail.
    with self.assertRaises(TypeError):
      _ = helpers.get_num_columns_of_2d_tensor([[1, 2], [3, 4]])

    # Trying to get the number of columns from a rank-1 tensor should fail.
    tensor = tf.convert_to_tensor([1, 2, 3, 4])
    with self.assertRaises(ValueError):
      _ = helpers.get_num_columns_of_2d_tensor(tensor)

    # Make sure that we successfully get the number of columns.
    tensor = tf.convert_to_tensor([[1, 2, 3]])
    self.assertEqual(3, helpers.get_num_columns_of_2d_tensor(tensor))


class UniqueListTest(tf.test.TestCase):
  """Tests for `UniqueList` classes."""

  def test_construct(self):
    """Tests the `UniqueList` constructor."""
    element1 = 1
    element2 = 2
    element3 = element1
    element4 = 4
    element5 = element4
    element6 = 6

    unique_list = helpers.UniqueList(
        [element1, element2, element3, element4, element5, element6])
    self.assertEqual(4, len(unique_list))
    self.assertEqual([element1, element2, element4, element6], unique_list.list)

  def test_append_raises(self):
    """Tests that "append" raises when given the wrong type."""
    unique_list = helpers.UniqueList(element_type=list)
    self.assertEqual(0, len(unique_list))
    self.assertEqual([], unique_list.list)

    with self.assertRaises(TypeError):
      # Since we passed element_type=list to the UniqueList constructor,
      # attempting to add any non-list should raise.
      unique_list.append(42)

  def test_add(self):
    """Tests `UniqueList`'s "__add__" method."""
    element1 = 1
    element2 = 2
    element3 = element1
    element4 = 4
    element5 = element4
    element6 = 6

    lhs = [element1, element2]
    rhs = [element3, element4, element5, element6]

    unique_list = helpers.UniqueList(lhs)
    self.assertEqual(2, len(unique_list))
    self.assertEqual([element1, element2], unique_list.list)

    unique_list += rhs
    self.assertEqual(4, len(unique_list))
    self.assertEqual([element1, element2, element4, element6], unique_list.list)

  def test_radd(self):
    """Tests `UniqueList`'s "__radd__" method."""
    element1 = 1
    element2 = 2
    element3 = element1
    element4 = 4
    element5 = element4
    element6 = 6

    lhs = [element1, element2]
    rhs = [element3, element4, element5, element6]

    unique_list = helpers.UniqueList(rhs)
    self.assertEqual(3, len(unique_list))
    self.assertEqual([element1, element4, element6], unique_list.list)

    unique_list = lhs + unique_list
    self.assertEqual(4, len(unique_list))
    self.assertEqual([element1, element2, element4, element6], unique_list.list)


if __name__ == "__main__":
  tf.test.main()
