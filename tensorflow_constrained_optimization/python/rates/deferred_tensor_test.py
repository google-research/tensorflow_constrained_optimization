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
"""Tests for deferred_tensor.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class DeferredTensorTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for `DeferredTensor` class."""

  def test_type_promotion(self):
    """Tests that automatic type promotion works as expected."""
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    tensor1 = deferred_tensor.ExplicitDeferredTensor(
        tf.constant(-2, dtype=tf.int16), auto_cast=True)
    tensor2 = deferred_tensor.ExplicitDeferredTensor(
        lambda: tf.constant(1.5, dtype=tf.float32), auto_cast=True)
    tensor3 = deferred_tensor.ExplicitDeferredTensor(
        tf.constant(2.7, dtype=tf.float32), auto_cast=True)
    tensor4 = deferred_tensor.ExplicitDeferredTensor(
        tf.constant(0.3, dtype=tf.float64), auto_cast=True)

    expression5 = tensor1 + tensor2
    expression6 = tensor3 / tensor4
    expression7 = expression5 * expression6

    value1 = tensor1(structure_memoizer)
    value2 = tensor2(structure_memoizer)
    value3 = tensor3(structure_memoizer)
    value4 = tensor4(structure_memoizer)
    value5 = expression5(structure_memoizer)
    value6 = expression6(structure_memoizer)
    value7 = expression7(structure_memoizer)

    self.assertEqual(tf.int16, value1.dtype.base_dtype)
    self.assertEqual(tf.float32, value2.dtype.base_dtype)
    self.assertEqual(tf.float32, value3.dtype.base_dtype)
    self.assertEqual(tf.float64, value4.dtype.base_dtype)
    self.assertEqual(tf.float32, value5.dtype.base_dtype)
    self.assertEqual(tf.float64, value6.dtype.base_dtype)
    self.assertEqual(tf.float64, value7.dtype.base_dtype)

    with self.wrapped_session() as session:
      self.assertAllClose(-2, session.run(value1))
      self.assertAllClose(1.5, session.run(value2))
      self.assertAllClose(2.7, session.run(value3))
      self.assertAllClose(0.3, session.run(value4))
      self.assertAllClose(-0.5, session.run(value5))
      self.assertAllClose(9, session.run(value6))
      self.assertAllClose(-4.5, session.run(value7))

  def test_callable(self):
    """Tests that callbacks are not called until needed."""
    # Keeps track of whether the callbacks have been called.
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    callback_list = []

    def callback1():
      callback_list.append("callback1")
      return 3.14

    def callback2():
      callback_list.append("callback2")
      return 4

    tensor1 = deferred_tensor.ExplicitDeferredTensor(callback1)
    tensor2 = deferred_tensor.ExplicitDeferredTensor(callback2)
    expression = tensor1 / tensor2

    # When we created the above expression, it should have created a closure,
    # instead of evaluating the arguments and performing the division.
    self.assertEmpty(callback_list)

    # We don't need to use a Session here, since the callbacks return scalars.
    self.assertAllEqual(0.785, expression(structure_memoizer))

    # Now that we've called expression(structure_memoizer), the callbacks should
    # each have been called once.
    self.assertAllEqual(["callback1", "callback2"], sorted(callback_list))

  def test_deferred_variable(self):
    """Tests that `DeferredVariable`s are created correctly."""
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    variable = deferred_tensor.DeferredVariable(42, dtype=tf.int32)

    # We should raise if we try to read a variable that hasn't been created.
    with self.assertRaises(RuntimeError):
      _ = variable(structure_memoizer)

    variable.create(structure_memoizer)

    # We should raise if we try to create the same variable a second time.
    with self.assertRaises(RuntimeError):
      variable.create(structure_memoizer)

    with self.wrapped_session() as session:
      self.assertAllEqual(42, session.run(variable(structure_memoizer)))

  def test_equality(self):
    """Tests that equal `DeferredTensor` are considered equal."""
    # Explicit DeferredTensors.
    tensor1 = deferred_tensor.ExplicitDeferredTensor([[1, 2], [-2, -1]])
    tensor2 = deferred_tensor.ExplicitDeferredTensor(
        np.array([[1.0, 2.0], [-2.0, -1.0]]))
    self.assertEqual(hash(tensor1), hash(tensor2))
    self.assertEqual(tensor1, tensor2)

    # Derived DeferredTensors.
    tensor3 = (tensor1 > 0)
    tensor4 = (tensor2 > 0)
    self.assertEqual(hash(tensor3), hash(tensor4))
    self.assertEqual(tensor3, tensor4)

  def test_non_equality(self):
    """Tests that unequal `DeferredTensor` are considered unequal."""
    # Explicit DeferredTensors.
    tensor1 = deferred_tensor.ExplicitDeferredTensor(
        np.array([[1.0, 2.0], [-2.0, -1.1]]))
    tensor2 = deferred_tensor.ExplicitDeferredTensor(
        np.array([[1.0, 2.0], [-2.0, -1.0]]))
    self.assertNotEqual(tensor1, tensor2)

    def zero(*args):
      del args
      return 0

    def one(*args):
      del args
      return 1

    # Derived DeferredTensors.
    tensor3 = deferred_tensor.DeferredTensor.apply(zero, tensor1, tensor2)
    tensor4 = deferred_tensor.DeferredTensor.apply(zero, tensor2, tensor1)
    tensor5 = deferred_tensor.DeferredTensor.apply(one, tensor1, tensor2)
    self.assertNotEqual(tensor3, tensor4)
    self.assertNotEqual(tensor4, tensor5)
    self.assertNotEqual(tensor3, tensor5)


if __name__ == "__main__":
  tf.test.main()
