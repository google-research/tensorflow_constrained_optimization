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
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.train import lagrangian_optimizer
from tensorflow_constrained_optimization.python.train import test_util
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class LagrangianOptimizerTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests the Lagrangian formulation and associated helper functions."""

  def _loss_test_helper(self,
                        minimization_problem,
                        expected_multipliers,
                        maximum_multiplier_radius=None):
    """Tests create_lagrangian_loss on the given minimization_problem."""
    loss_fn, update_ops_fn, state_variable = (
        lagrangian_optimizer.create_lagrangian_loss(
            minimization_problem,
            maximum_multiplier_radius=maximum_multiplier_radius))
    optimizer = tf.keras.optimizers.SGD(1.0)
    var_list = minimization_problem.trainable_variables + [state_variable]

    if tf.executing_eagerly():
      train_op_fn = lambda: optimizer.minimize(loss_fn, var_list)
    else:
      # If we're in graph mode, then we need to create the train_op before the
      # session, so that we know which variables need to be initialized.
      train_op = optimizer.minimize(loss_fn, var_list)
      train_op_fn = lambda: train_op

    # Perform a few steps of training, and record the Lagrange multipliers.
    states = []
    with self.wrapped_session() as session:
      while len(states) < len(expected_multipliers):
        session.run_ops(update_ops_fn)
        states.append(session.run(state_variable))
        session.run_ops(train_op_fn)

    # Compare the observed sequence of Lagrange multipliers to the expected one.
    for expected, actual in zip(expected_multipliers, states):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)

  def _optimizer_v1_test_helper(self,
                                minimization_problem,
                                expected_multipliers,
                                maximum_multiplier_radius=None):
    """Tests LagrangianOptimizerV1 on the given minimization_problem."""
    # The "optimizer" argument won't actually be used, here, but we provide two
    # different optimizers to make sure that proxy-Lagrangian state updates are
    # correctly dispatched to the "constraint_optimizer".
    #
    # Also notice that we force the Lagrange multipliers to be created here by
    # providing the "num_constraints" argument, so that (1) in graph mode,
    # they'll be initialized correctly and (2) in eager mode, we'll be able to
    # check their initial values.
    optimizer = lagrangian_optimizer.LagrangianOptimizerV1(
        optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.1),
        num_constraints=minimization_problem.num_constraints,
        constraint_optimizer=tf.compat.v1.train.GradientDescentOptimizer(1.0),
        maximum_multiplier_radius=maximum_multiplier_radius)

    if tf.executing_eagerly():
      train_op_fn = lambda: optimizer.minimize(minimization_problem)
    else:
      # If we're in graph mode, then we need to create the train_op before the
      # session, so that we know which variables need to be initialized.
      train_op = optimizer.minimize(minimization_problem)
      train_op_fn = lambda: train_op

    # Perform a few steps of training, and record the Lagrange multipliers.
    states = []
    with self.wrapped_session() as session:
      while len(states) < len(expected_multipliers):
        states.append(session.run(optimizer._formulation.state))
        session.run_ops(train_op_fn)

    # Compare the observed sequence of Lagrange multipliers to the expected one.
    for expected, actual in zip(expected_multipliers, states):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)

  def _optimizer_v2_test_helper(self,
                                minimization_problem,
                                expected_multipliers,
                                maximum_multiplier_radius=None):
    """Tests LagrangianOptimizerV2 on the given minimization_problem."""
    # The "optimizer" argument won't actually be used, here, but we provide two
    # different optimizers to make sure that proxy-Lagrangian state updates are
    # correctly dispatched to the "constraint_optimizer".
    optimizer = lagrangian_optimizer.LagrangianOptimizerV2(
        optimizer=tf.keras.optimizers.SGD(0.1),
        num_constraints=minimization_problem.num_constraints,
        constraint_optimizer=tf.keras.optimizers.SGD(1.0),
        maximum_multiplier_radius=maximum_multiplier_radius)
    var_list = (
        minimization_problem.trainable_variables +
        optimizer.trainable_variables())

    if tf.executing_eagerly():
      train_op_fn = lambda: optimizer.minimize(minimization_problem, var_list)
    else:
      # If we're in graph mode, then we need to create the train_op before the
      # session, so that we know which variables need to be initialized.
      train_op = optimizer.minimize(minimization_problem, var_list)
      train_op_fn = lambda: train_op

    # Perform a few steps of training, and record the Lagrange multipliers.
    states = []
    with self.wrapped_session() as session:
      while len(states) < len(expected_multipliers):
        states.append(session.run(optimizer._formulation.state))
        session.run_ops(train_op_fn)

    # Compare the observed sequence of Lagrange multipliers to the expected one.
    for expected, actual in zip(expected_multipliers, states):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)

  def _test_helper(self,
                   minimization_problem,
                   expected_multipliers,
                   maximum_multiplier_radius=None):
    self._loss_test_helper(
        minimization_problem,
        expected_multipliers,
        maximum_multiplier_radius=maximum_multiplier_radius)
    self._optimizer_v1_test_helper(
        minimization_problem,
        expected_multipliers,
        maximum_multiplier_radius=maximum_multiplier_radius)
    self._optimizer_v2_test_helper(
        minimization_problem,
        expected_multipliers,
        maximum_multiplier_radius=maximum_multiplier_radius)

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

    expected_multipliers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.6, 0.0, 0.4]),
        np.array([0.7, 0.0, 0.3]),
        np.array([0.8, 0.0, 0.2]),
        np.array([0.9, 0.0, 0.1]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]

    self._test_helper(
        minimization_problem,
        expected_multipliers,
        maximum_multiplier_radius=1.0)


if __name__ == "__main__":
  tf.test.main()
