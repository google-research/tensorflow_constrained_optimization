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
"""Tests for loss.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import loss
# Placeholder for internal import.


def binary_zero_one_loss(predictions):
  """Expected binary classification zero-one loss."""
  return (np.sign(predictions) + 1.0) / 2.0


def binary_hinge_loss(margin):
  """Expected binary classification hinge loss with given margin."""
  return lambda predictions: np.maximum(0.0, margin + predictions)


def binary_softmax_loss(predictions):
  """Expected binary classification softmax loss."""
  numerator = np.exp(predictions)
  return numerator / (1 + numerator)


def binary_softmax_cross_entropy_loss(predictions):
  """Expected binary classification softmax cross-entropy loss."""
  return np.log(1 + np.exp(predictions))


# @run_all_tests_in_graph_and_eager_modes
class LossTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for `Loss` classes."""

  def __init__(self, *args, **kwargs):
    super(LossTest, self).__init__(*args, **kwargs)

    # We use a fixed fake dataset to make sure that the tests are reproducible.
    # The code for generating this random dataset is:
    #
    #   size = 20
    #
    #   self._predictions = np.random.randn(size)
    #   self._weights = np.random.randn(size, 2)
    #
    # The dataset itself is:
    self._predictions = np.array([
        0.584512341999, -0.55015387642, 0.0903356546458, 0.633426692118,
        -0.882615385953, -0.3678706889, 1.04663484707, 0.0344508924989,
        0.216541984044, -0.717996402915, -0.600528960814, -0.337885591055,
        -0.49071694553, 0.446409688644, 0.113508013003, -2.28022263467,
        -1.74425324236, -0.0132548715691, -0.923112276794, 1.11681572053
    ])
    self._weights = np.array([
        [-0.452099600241, 1.07725936148],
        [0.703368020437, -1.4504981454],
        [-0.0980201564317, 0.92831478234],
        [0.258419891287, -2.09698564434],
        [0.0658564429054, 1.41984871297],
        [-0.788784248269, -1.18654649943],
        [-1.73866936442, -0.0590830456781],
        [-0.23615112839, -1.75620893231],
        [0.676139218764, -0.0408724788464],
        [-0.618938863154, -0.296080582242],
        [-2.25835513009, -0.0213691762614],
        [0.0766053894937, -1.13339177424],
        [2.36519239138, 0.681535742442],
        [-0.229637659991, -2.26443855249],
        [0.171883509387, -0.112264766436],
        [-0.499339463268, 0.202125957361],
        [-0.134112250792, -0.355461297829],
        [0.608786974973, -0.324075441957],
        [0.814526800499, 2.17971326555],
        [-0.665005832144, 1.76628797184],
    ])

  def _binary_loss_helper(self, expected_loss, actual_loss):
    minimum_weights = np.minimum(self._weights[:, 0], self._weights[:, 1])
    expected = minimum_weights + (
        expected_loss(self._predictions) *
        (self._weights[:, 0] - minimum_weights) +
        expected_loss(-self._predictions) *
        (self._weights[:, 1] - minimum_weights))

    with self.wrapped_session() as session:
      actual = session.run(
          actual_loss.evaluate_binary_classification(
              tf.constant(self._predictions), tf.constant(self._weights)))

    self.assertAllClose(expected, actual, rtol=0, atol=1e-6)

  def test_convert_to_binary_classification_predictions(self):
    """Tests the "_convert_to_binary_classification_predictions" function."""
    # A non-Tensor isn't a valid set of binary predictions.
    with self.assertRaises(TypeError):
      _ = loss._convert_to_binary_classification_predictions(
          [1.0, 2.0, 3.0, 4.0])

    # An integer Tensor isn't a valid set of binary predictions.
    tensor = tf.convert_to_tensor([1, 2, 3, 4], dtype=tf.int32)
    with self.assertRaises(TypeError):
      _ = loss._convert_to_binary_classification_predictions(tensor)

    # A rank-2 Tensor isn't a valid set of binary predictions.
    tensor = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    with self.assertRaises(ValueError):
      _ = loss._convert_to_binary_classification_predictions(tensor)

    # A floating-point Tensor of shape (1,4,1) is a valid set of binary
    # predictions (it's floating-point, and only has one "nontrivial"
    # dimension).
    expected_predictions = [1.0, 2.0, 3.0, 4.0]
    tensor = tf.convert_to_tensor([[[1.0], [2.0], [3.0], [4.0]]],
                                  dtype=tf.float32)
    with self.wrapped_session() as session:
      actual_predictions = session.run(
          loss._convert_to_binary_classification_predictions(tensor))
    self.assertAllClose(
        expected_predictions, actual_predictions, rtol=0, atol=1e-6)

  def test_binary_zero_one_loss(self):
    self._binary_loss_helper(binary_zero_one_loss, loss.ZeroOneLoss())

  def test_binary_hinge_loss(self):
    for margin in [0.5, 1.0, 1.3]:
      self._binary_loss_helper(
          binary_hinge_loss(margin), loss.HingeLoss(margin))

  def test_binary_softmax_loss(self):
    self._binary_loss_helper(binary_softmax_loss, loss.SoftmaxLoss())

  def test_binary_softmax_cross_entropy_loss(self):
    self._binary_loss_helper(binary_softmax_cross_entropy_loss,
                             loss.SoftmaxCrossEntropyLoss())


if __name__ == "__main__":
  tf.test.main()
