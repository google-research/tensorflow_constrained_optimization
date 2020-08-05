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
"""Tests for keras.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import keras
from tensorflow_constrained_optimization.python.rates import operations
# Placeholder for internal import.


class MockKerasLayer(keras.KerasLayer):

  def __init__(self, *args, **kwargs):
    self._names = []
    # We need to create self._names before calling the KerasLayer's __init__
    # method since it will call _variable_fn.
    super(MockKerasLayer, self).__init__(*args, **kwargs)

  @property
  def names(self):
    return self._names[:]

  def _variable_fn(self, *args, name=None, **kwargs):
    self._names.append(name)
    return super(MockKerasLayer, self)._variable_fn(*args, name=name, **kwargs)


# @run_all_tests_in_graph_and_eager_modes
class KerasTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for classes in keras.py."""

  def test_keras_placeholder(self):
    """Tests the `KerasPlaceholder` class."""
    y_pred_placeholder = keras.KerasPlaceholder(lambda y_true, y_pred: y_pred)
    y_true_placeholder = keras.KerasPlaceholder(lambda y_true, y_pred: y_true)

    y_pred_placeholder.assign(-2.0, -1.0)
    y_true_placeholder.assign(1.0, 2.0)
    self.assertEqual(-1.0, y_pred_placeholder())
    self.assertEqual(1.0, y_true_placeholder())

    y_pred_placeholder.assign()
    y_true_placeholder.assign()
    with self.assertRaises(RuntimeError):
      _ = y_pred_placeholder()
    with self.assertRaises(RuntimeError):
      _ = y_true_placeholder()

  def test_keras_layer(self):
    """Tests the `KerasLayer` class."""
    zero_placeholder = keras.KerasPlaceholder(
        lambda y_true, y_pred: tf.constant(0.0))
    one_constant = lambda: tf.constant(1.0)
    two_placeholder = keras.KerasPlaceholder(
        lambda y_true, y_pred: tf.constant(0.2))
    three_constant = lambda: tf.constant(0.03)
    four_placeholder = keras.KerasPlaceholder(
        lambda y_true, y_pred: tf.constant(0.004))

    zero = operations.wrap_rate(zero_placeholder)
    one = operations.wrap_rate(one_constant)
    two = operations.wrap_rate(two_placeholder)
    three = operations.wrap_rate(three_constant)
    four = operations.wrap_rate(four_placeholder)

    objective = sum([one, two, three, four], zero)

    # Trying to create a KerasLayer without including the input placeholders
    # should raise.
    with self.assertRaises(ValueError):
      _ = keras.KerasLayer(objective=objective)

    placeholders = [zero_placeholder, two_placeholder, four_placeholder]
    layer = MockKerasLayer(objective=objective, placeholders=placeholders)

    # Make sure that we created the variables we expected.
    expected_names = {"tfco_global_step", "tfco_lagrange_multipliers"}
    actual_names = set(layer.names)
    self.assertEqual(expected_names, actual_names)
    # and that the layer is aware of them.
    keras_names = [weight.name for weight in layer.weights]
    keras_names.sort()
    self.assertTrue(keras_names[0].startswith("tfco_global_step"))
    self.assertTrue(keras_names[1].startswith("tfco_lagrange_multipliers"))

    # Make sure that the loss evaluates to the correct value, including both the
    # constant and placeholder inputs.
    with self.wrapped_session() as session:
      result = session.run(layer.loss(-10.0, -100.0))
    self.assertNear(1.234, result, err=1e-6)


if __name__ == "__main__":
  tf.test.main()
