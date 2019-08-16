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
"""Tests for lagrangian_optimizer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.train import lagrangian_optimizer
from tensorflow_constrained_optimization.python.train import test_util


# @tf.contrib.eager.run_all_tests_in_graph_and_eager_modes
class LagrangianOptimizerTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests the `LagrangianOptimizer` and associated helper functions."""

  def test_project_multipliers_wrt_euclidean_norm(self):
    """Tests Euclidean projection routine on some known values."""
    multipliers1 = tf.constant([-0.1, -0.6, -0.3])
    expected_projected_multipliers1 = np.array([0.0, 0.0, 0.0])

    multipliers2 = tf.constant([-0.1, 0.6, 0.3])
    expected_projected_multipliers2 = np.array([0.0, 0.6, 0.3])

    multipliers3 = tf.constant([0.4, 0.7, -0.2, 0.5, 0.1])
    expected_projected_multipliers3 = np.array([0.2, 0.5, 0.0, 0.3, 0.0])

    with self.wrapped_session() as session:
      projected_multipliers1 = session.run(
          lagrangian_optimizer._project_multipliers_wrt_euclidean_norm(
              multipliers1, 1.0))
      projected_multipliers2 = session.run(
          lagrangian_optimizer._project_multipliers_wrt_euclidean_norm(
              multipliers2, 1.0))
      projected_multipliers3 = session.run(
          lagrangian_optimizer._project_multipliers_wrt_euclidean_norm(
              multipliers3, 1.0))

    self.assertAllClose(
        expected_projected_multipliers1,
        projected_multipliers1,
        rtol=0,
        atol=1e-6)
    self.assertAllClose(
        expected_projected_multipliers2,
        projected_multipliers2,
        rtol=0,
        atol=1e-6)
    self.assertAllClose(
        expected_projected_multipliers3,
        projected_multipliers3,
        rtol=0,
        atol=1e-6)

  def test_lagrangian_optimizer(self):
    """Tests that the Lagrange multipliers update as expected."""
    minimization_problem = test_util.ConstantMinimizationProblem(
        np.array([0.6, -0.1, 0.4]))
    optimizer = lagrangian_optimizer.LagrangianOptimizer(
        tf.train.GradientDescentOptimizer(1.0), maximum_multiplier_radius=1.0)
    # We force the Lagrange multipliers to be created here so that (1) in
    # graph mode, it will be initialized correctly and (2) in eager mode,
    # we'll be able to check its initial value.
    optimizer._maybe_create_multipliers(minimization_problem.num_constraints)

    expected_multipliers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.6, 0.0, 0.4]),
        np.array([0.7, 0.0, 0.3]),
        np.array([0.8, 0.0, 0.2]),
        np.array([0.9, 0.0, 0.1]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]

    multipliers = []
    with self.wrapped_session() as session:
      while len(multipliers) < len(expected_multipliers):
        multipliers.append(session.run(optimizer._multipliers))
        session.run(optimizer.minimize(minimization_problem))

    for expected, actual in zip(expected_multipliers, multipliers):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)

  def test_lagrangian_loss(self):
    """Tests that the Lagrange multipliers update as expected."""
    minimization_problem = test_util.ConstantMinimizationProblem(
        np.array([0.6, -0.1, 0.4]))
    loss_fn, pre_train_ops_fn, multipliers_variable = (
        lagrangian_optimizer.create_lagrangian_loss(
            minimization_problem, maximum_multiplier_radius=1.0))
    optimizer = tf.train.GradientDescentOptimizer(1.0)

    loss = loss_fn
    if not tf.executing_eagerly():
      loss = loss()

    expected_multipliers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.6, 0.0, 0.4]),
        np.array([0.7, 0.0, 0.3]),
        np.array([0.8, 0.0, 0.2]),
        np.array([0.9, 0.0, 0.1]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]

    multipliers = []
    with self.wrapped_session() as session:
      while len(multipliers) < len(expected_multipliers):
        session.run_ops(pre_train_ops_fn)
        multipliers.append(session.run(multipliers_variable))
        session.run_ops(lambda: optimizer.minimize(loss))

    for expected, actual in zip(expected_multipliers, multipliers):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
