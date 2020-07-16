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
import tensorflow.compat.v2 as tf

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


def multiclass_zero_one_loss(predictions, labels):
  """Expected multiclass zero-one loss."""
  maximum_predictions = np.amax(predictions, axis=1, keepdims=True)
  zero_one_predictions = (predictions >= maximum_predictions).astype(np.float32)
  zero_one_predictions /= np.sum(zero_one_predictions, axis=1, keepdims=True)
  return (1.0 - zero_one_predictions[range(len(labels)), labels])


def multiclass_hinge_loss(margin):
  """Expected multiclass hinge loss with given margin."""

  def loss_fn(predictions, labels):
    true_predictions = predictions[range(len(labels)), labels]

    incorrect_predictions = np.copy(predictions)
    incorrect_predictions[range(len(labels)), labels] = -float("Inf")
    false_predictions = np.amax(incorrect_predictions, axis=1)

    return np.maximum(0.0, margin + false_predictions - true_predictions)

  return loss_fn


def multiclass_softmax_loss(predictions, labels):
  """Expected multiclass softmax loss."""
  numerator = np.exp(predictions)
  softmax_predictions = numerator / np.sum(numerator, axis=1, keepdims=True)
  return (1.0 - softmax_predictions[range(len(labels)), labels])


def multiclass_softmax_cross_entropy_loss(predictions, labels):
  """Expected multiclass softmax cross-entropy loss."""
  numerator = np.exp(predictions)
  softmax_predictions = numerator / np.sum(numerator, axis=1, keepdims=True)
  return -np.log(softmax_predictions[range(len(labels)), labels])


# @run_all_tests_in_graph_and_eager_modes
class LossTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for `Loss` classes."""

  def __init__(self, *args, **kwargs):
    super(LossTest, self).__init__(*args, **kwargs)

    # We use a fixed fake dataset to make sure that the tests are reproducible.
    # The code for generating this random dataset is:
    #
    #   self._size = 20
    #   self._num_classes = 4
    #
    #   self._binary_predictions = np.random.randn(self._size)
    #   self._binary_weights = np.random.randn(self._size, 2)
    #   self._multiclass_predictions = np.random.randn(self._size
    #                                                  self._num_classes)
    #   self._multiclass_labels = np.random.randint(self._num_classes,
    #                                               size=self._size)
    self._size = 20
    self._num_classes = 4

    self._binary_predictions = np.array([
        0.584512341999, -0.55015387642, 0.0903356546458, 0.633426692118,
        -0.882615385953, -0.3678706889, 1.04663484707, 0.0344508924989,
        0.216541984044, -0.717996402915, -0.600528960814, -0.337885591055,
        -0.49071694553, 0.446409688644, 0.113508013003, -2.28022263467,
        -1.74425324236, -0.0132548715691, -0.923112276794, 1.11681572053
    ])
    self._binary_weights = np.array([
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

    self._multiclass_predictions = np.array(
        [[-2.02659328, 0.33254821, 0.63735689, -1.20328286],
         [-1.91117858, -0.8828284, -0.13994554, -1.40856467],
         [-0.30937745, 0.03708433, 0.43777548, 0.38128008],
         [-0.99757322, -0.16391412, -0.59515243, -0.68399455],
         [-0.3835558, -0.39887864, -0.09352329, -0.00960708],
         [-0.20833296, -2.09496521, -0.92875963, 0.36322199],
         [1.38877123, -1.00741888, -0.6697507, -0.44300706],
         [0.45140316, -0.34186604, 1.16478144, 1.55422841],
         [1.29535797, 0.30848108, -0.79274869, 0.06619779],
         [-1.42204585, -0.8870909, -1.18777075, -0.04280105],
         [-0.43520423, -1.28295244, 1.04493067, 0.88954064],
         [0.46216183, -0.57945345, 0.35992645, -0.53958252],
         [0.48783054, -0.53686745, -0.13646917, 1.66356563],
         [-0.44541233, 1.80394394, -1.94286545, 1.62578299],
         [0.81490563, -0.87840684, 1.95922503, -0.07449408],
         [-2.31883821, 1.03015801, 0.90836021, 0.92499435],
         [0.45671034, 0.72803667, -0.46070285, 0.46731283],
         [0.67588137, 1.11618973, 0.70730887, -1.3699823],
         [0.26233944, -0.80859112, 0.64678011, -1.21484718],
         [0.41000457, 1.09239098, -0.63955167, -0.75230251]])
    self._multiclass_labels = np.array(
        [1, 2, 3, 0, 2, 3, 3, 2, 2, 0, 1, 1, 1, 2, 3, 2, 0, 3, 1, 1])

  def _binary_loss_helper(self, expected_binary_loss, actual_loss):
    # We first make sure that the multiclass loss generalizes the binary
    # classification loss.
    minimum_weights = np.minimum(self._binary_weights[:, 0],
                                 self._binary_weights[:, 1])
    expected = minimum_weights + (
        expected_binary_loss(self._binary_predictions) *
        (self._binary_weights[:, 0] - minimum_weights) +
        expected_binary_loss(-self._binary_predictions) *
        (self._binary_weights[:, 1] - minimum_weights))

    predictions = tf.constant(self._binary_predictions)
    weights = tf.constant(self._binary_weights)
    with self.wrapped_session() as session:
      actual_binary = session.run(
          actual_loss.evaluate_binary_classification(predictions, weights))
      actual_multiclass = session.run(
          actual_loss.evaluate_multiclass(
              tf.stack([0.5 * predictions, -0.5 * predictions], axis=1),
              weights))

    self.assertAllClose(expected, actual_binary, rtol=0, atol=1e-6)
    self.assertAllClose(expected, actual_multiclass, rtol=0, atol=1e-6)

  def _multiclass_loss_helper(self, expected_multiclass_loss, actual_loss):
    # Tests that the multiclass loss generalizes the usual multiclass loss (our
    # losses differ in that we can have a different cost for each class, instead
    # of only having a single label).
    expected = expected_multiclass_loss(self._multiclass_predictions,
                                        self._multiclass_labels)

    # We only have labels, but we need per-example-per-class weights.
    multiclass_weights = np.ones((self._size, self._num_classes))
    multiclass_weights[range(self._size), self._multiclass_labels] = 0.0

    predictions = tf.constant(self._multiclass_predictions)
    weights = tf.constant(multiclass_weights)
    with self.wrapped_session() as session:
      actual = session.run(
          actual_loss.evaluate_multiclass(predictions, weights))

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

  def test_zero_one_loss(self):
    self._binary_loss_helper(binary_zero_one_loss, loss.ZeroOneLoss())
    self._multiclass_loss_helper(multiclass_zero_one_loss, loss.ZeroOneLoss())

  def test_hinge_loss(self):
    for margin in [0.5, 1.0, 1.3]:
      self._binary_loss_helper(
          binary_hinge_loss(margin), loss.HingeLoss(margin))
      self._multiclass_loss_helper(
          multiclass_hinge_loss(margin), loss.HingeLoss(margin))

  def test_softmax_loss(self):
    self._binary_loss_helper(binary_softmax_loss, loss.SoftmaxLoss())
    self._multiclass_loss_helper(multiclass_softmax_loss, loss.SoftmaxLoss())

  def test_softmax_cross_entropy_loss(self):
    self._binary_loss_helper(binary_softmax_cross_entropy_loss,
                             loss.SoftmaxCrossEntropyLoss())
    self._multiclass_loss_helper(multiclass_softmax_cross_entropy_loss,
                                 loss.SoftmaxCrossEntropyLoss())


if __name__ == "__main__":
  tf.test.main()
