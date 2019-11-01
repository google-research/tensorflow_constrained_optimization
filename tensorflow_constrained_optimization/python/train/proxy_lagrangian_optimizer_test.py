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
"""Tests for proxy_lagrangian_optimizer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.train import proxy_lagrangian_optimizer
from tensorflow_constrained_optimization.python.train import test_util
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class ProxyLagrangianOptimizerTest(
    graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests proxy-Lagrangian formulation and associated helper functions."""

  def _loss_test_helper(self,
                        minimization_problem,
                        expected_states,
                        regret_type,
                        update_type,
                        minimum_multiplier_radius=None,
                        initial_multiplier_radius=None):
    """Tests create_proxy_lagrangian_loss on the given minimization_problem."""
    loss_fn, update_ops_fn, state_variable = (
        proxy_lagrangian_optimizer.create_proxy_lagrangian_loss(
            minimization_problem,
            regret_type=regret_type,
            update_type=update_type,
            minimum_multiplier_radius=minimum_multiplier_radius,
            initial_multiplier_radius=initial_multiplier_radius))
    optimizer = tf.keras.optimizers.SGD(1.0)
    var_list = minimization_problem.trainable_variables + [state_variable]

    if tf.executing_eagerly():
      train_op_fn = lambda: optimizer.minimize(loss_fn, var_list)
    else:
      # If we're in graph mode, then we need to create the train_op before the
      # session, so that we know which variables need to be initialized.
      train_op = optimizer.minimize(loss_fn, var_list)
      train_op_fn = lambda: train_op

    # Perform a few steps of training, and record the proxy-Lagrangian state.
    states = []
    with self.wrapped_session() as session:
      while len(states) < len(expected_states):
        session.run_ops(update_ops_fn)
        states.append(session.run(state_variable))
        session.run_ops(train_op_fn)

    # Compare the observed sequence of proxy-Lagrangian states to the expected
    # one.
    for expected, actual in zip(expected_states, states):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)

  def _optimizer_v1_test_helper(self,
                                minimization_problem,
                                expected_states,
                                regret_type,
                                update_type,
                                minimum_multiplier_radius=None,
                                initial_multiplier_radius=None):
    """Tests ProxyLagrangianOptimizerV1 on the given minimization_problem."""
    # The "optimizer" argument won't actually be used, here, but we provide two
    # different optimizers to make sure that proxy-Lagrangian state updates are
    # correctly dispatched to the "constraint_optimizer".
    #
    # Also notice that we force the internal state to be created here by
    # providing the "num_constraints" argument, so that (1) in graph mode, it
    # will be initialized correctly and (2) in eager mode, we'll be able to
    # check its initial value.
    optimizer = proxy_lagrangian_optimizer.ProxyLagrangianOptimizerV1(
        optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.1),
        num_constraints=minimization_problem.num_constraints,
        constraint_optimizer=tf.compat.v1.train.GradientDescentOptimizer(1.0),
        regret_type=regret_type,
        update_type=update_type,
        minimum_multiplier_radius=minimum_multiplier_radius,
        initial_multiplier_radius=initial_multiplier_radius)

    if tf.executing_eagerly():
      train_op_fn = lambda: optimizer.minimize(minimization_problem)
    else:
      # If we're in graph mode, then we need to create the train_op before the
      # session, so that we know which variables need to be initialized.
      train_op = optimizer.minimize(minimization_problem)
      train_op_fn = lambda: train_op

    # Perform a few steps of training, and record the proxy-Lagrangian state.
    states = []
    with self.wrapped_session() as session:
      while len(states) < len(expected_states):
        states.append(session.run(optimizer._formulation.state))
        session.run_ops(train_op_fn)

    # Compare the observed sequence of proxy-Lagrangian states to the expected
    # one.
    for expected, actual in zip(expected_states, states):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)

  def _optimizer_v2_test_helper(self,
                                minimization_problem,
                                expected_states,
                                regret_type,
                                update_type,
                                minimum_multiplier_radius=None,
                                initial_multiplier_radius=None):
    """Tests ProxyLagrangianOptimizerV2 on the given minimization_problem."""
    # The "optimizer" argument won't actually be used, here, but we provide two
    # different optimizers to make sure that proxy-Lagrangian state updates are
    # correctly dispatched to the "constraint_optimizer".
    optimizer = proxy_lagrangian_optimizer.ProxyLagrangianOptimizerV2(
        optimizer=tf.keras.optimizers.SGD(0.1),
        num_constraints=minimization_problem.num_constraints,
        constraint_optimizer=tf.keras.optimizers.SGD(1.0),
        regret_type=regret_type,
        update_type=update_type,
        minimum_multiplier_radius=minimum_multiplier_radius,
        initial_multiplier_radius=initial_multiplier_radius)
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

    # Perform a few steps of training, and record the proxy-Lagrangian state.
    states = []
    with self.wrapped_session() as session:
      while len(states) < len(expected_states):
        states.append(session.run(optimizer._formulation.state))
        session.run_ops(train_op_fn)

    # Compare the observed sequence of proxy-Lagrangian states to the expected
    # one.
    for expected, actual in zip(expected_states, states):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)

  def _test_helper(self,
                   minimization_problem,
                   expected_states,
                   regret_type,
                   update_type,
                   minimum_multiplier_radius=None,
                   initial_multiplier_radius=None):
    self._loss_test_helper(
        minimization_problem,
        expected_states,
        regret_type=regret_type,
        update_type=update_type,
        minimum_multiplier_radius=minimum_multiplier_radius,
        initial_multiplier_radius=initial_multiplier_radius)
    self._optimizer_v1_test_helper(
        minimization_problem,
        expected_states,
        regret_type=regret_type,
        update_type=update_type,
        minimum_multiplier_radius=minimum_multiplier_radius,
        initial_multiplier_radius=initial_multiplier_radius)
    self._optimizer_v2_test_helper(
        minimization_problem,
        expected_states,
        regret_type=regret_type,
        update_type=update_type,
        minimum_multiplier_radius=minimum_multiplier_radius,
        initial_multiplier_radius=initial_multiplier_radius)

  def test_maximum_eigenvector_power_method(self):
    """Tests power method routine on some known left-stochastic matrices."""
    matrix1 = np.matrix([[0.6, 0.1, 0.1], [0.0, 0.6, 0.9], [0.4, 0.3, 0.0]])
    matrix2 = np.matrix([[0.4, 0.4, 0.2], [0.2, 0.1, 0.5], [0.4, 0.5, 0.3]])

    with self.wrapped_session() as session:
      eigenvector1 = session.run(
          proxy_lagrangian_optimizer._maximal_eigenvector_power_method(
              tf.constant(matrix1)))
      eigenvector2 = session.run(
          proxy_lagrangian_optimizer._maximal_eigenvector_power_method(
              tf.constant(matrix2)))

    # Check that eigenvector1 and eigenvector2 are eigenvectors of matrix1 and
    # matrix2 (respectively) with associated eigenvalue 1.
    matrix_eigenvector1 = np.tensordot(matrix1, eigenvector1, axes=1)
    matrix_eigenvector2 = np.tensordot(matrix2, eigenvector2, axes=1)
    self.assertAllClose(eigenvector1, matrix_eigenvector1, rtol=0, atol=1e-6)
    self.assertAllClose(eigenvector2, matrix_eigenvector2, rtol=0, atol=1e-6)

  def test_project_distribution_wrt_euclidean_norm(self):
    """Tests Euclidean projection routine on some known values."""
    distributions = [
        tf.constant([-0.1, -0.8, -0.3]),
        tf.constant([-0.1, 0.4, 0.1]),
        tf.constant([0.4, 1.2, 0.2])
    ]
    expected_projected_distributions = [
        np.array([0.6, 0.0, 0.4]),
        np.array([0.1, 0.6, 0.3]),
        np.array([0.1, 0.9, 0.0])
    ]

    projected_distributions = []
    with self.wrapped_session() as session:
      for distribution in distributions:
        projected_distributions.append(
            session.run(
                proxy_lagrangian_optimizer
                ._project_distribution_wrt_euclidean_norm(distribution)))

    self.assertAllClose(
        expected_projected_distributions,
        projected_distributions,
        rtol=0,
        atol=1e-6)

  def test_project_distribution_wrt_euclidean_norm_on_matrix(self):
    """Tests Euclidean projection routine on some known values."""
    matrix = tf.constant([[-0.1, -0.1, 0.4], [-0.8, 0.4, 1.2], [-0.3, 0.1,
                                                                0.2]])
    expected_projected_matrix = np.array([[0.6, 0.1, 0.1], [0.0, 0.6, 0.9],
                                          [0.4, 0.3, 0.0]])

    with self.wrapped_session() as session:
      projected_matrix = session.run(
          proxy_lagrangian_optimizer._project_distribution_wrt_euclidean_norm(
              matrix))

    self.assertAllClose(
        expected_projected_matrix, projected_matrix, rtol=0, atol=1e-6)

  def test_project_log_distribution_wrt_kl_divergence(self):
    """Tests KL-divergence projection routine on some known values."""
    distributions = [
        tf.constant([0.2, 0.1, 0.2]),
        tf.constant([0.8, 0.2, 1.0]),
        tf.constant([0.6, 1.5, 0.9])
    ]
    expected_projected_distributions = [
        np.array([0.4, 0.2, 0.4]),
        np.array([0.4, 0.1, 0.5]),
        np.array([0.2, 0.5, 0.3])
    ]

    projected_distributions = []
    with self.wrapped_session() as session:
      for distribution in distributions:
        projected_distributions.append(
            session.run(
                tf.exp(
                    proxy_lagrangian_optimizer
                    ._project_log_distribution_wrt_kl_divergence(
                        tf.math.log(distribution)))))

    self.assertAllClose(
        expected_projected_distributions,
        projected_distributions,
        rtol=0,
        atol=1e-6)

  def test_project_log_distribution_wrt_kl_divergence_on_matrix(self):
    """Tests KL-divergence projection routine on some known values."""
    matrix = tf.constant([[0.2, 0.8, 0.6], [0.1, 0.2, 1.5], [0.2, 1.0, 0.9]])
    expected_projected_matrix = np.array([[0.4, 0.4, 0.2], [0.2, 0.1, 0.5],
                                          [0.4, 0.5, 0.3]])

    with self.wrapped_session() as session:
      projected_matrix = session.run(
          tf.exp(
              proxy_lagrangian_optimizer
              ._project_log_distribution_wrt_kl_divergence(
                  tf.math.log(matrix))))

    self.assertAllClose(
        expected_projected_matrix, projected_matrix, rtol=0, atol=1e-6)

  def test_additive_external_proxy_lagrangian_optimizer(self):
    """Tests that the distributions update as expected."""
    minimization_problem = test_util.ConstantMinimizationProblem(
        np.array([0.6, -0.1, 0.4]))
    # Calculated using a numpy+python implementation of the algorithm.
    expected_states = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.66666667, 0.26666667, 0.0, 0.06666667]),
        np.array([0.33333333, 0.53333333, 0.0, 0.13333333]),
    ]

    self._test_helper(
        minimization_problem,
        expected_states,
        regret_type="external",
        update_type="additive")

  def test_multiplicative_external_proxy_lagrangian_optimizer(self):
    """Tests that the distributions update as expected."""
    minimization_problem = test_util.ConstantMinimizationProblem(
        np.array([0.6, -0.1, 0.4]))
    # Calculated using a numpy+python implementation of the algorithm.
    expected_states = [
        np.log(np.array([0.4, 0.2, 0.2, 0.2])),
        np.log(np.array([0.32160644, 0.29300257, 0.14550077, 0.23989022])),
        np.log(np.array([0.23910893, 0.3969348, 0.09788292, 0.26607335])),
    ]

    self._test_helper(
        minimization_problem,
        expected_states,
        regret_type="external",
        update_type="multiplicative",
        initial_multiplier_radius=0.8)

  def test_additive_swap_proxy_lagrangian_optimizer(self):
    """Tests that the stochastic matrices update as expected."""
    minimization_problem = test_util.ConstantMinimizationProblem(
        np.array([0.6, -0.1, 0.4]))
    # Calculated using a numpy+python implementation of the algorithm.
    expected_states = [
        np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        np.array([[0.66666667, 1.0, 1.0, 1.0], [0.26666667, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0], [0.06666667, 0.0, 0.0, 0.0]]),
        np.array([[0.41666667, 0.93333333, 1.0, 0.98333333],
                  [0.46666667, 0.05333333, 0.0,
                   0.01333333], [0.0, 0.0, 0.0, 0.0],
                  [0.11666667, 0.01333333, 0.0, 0.00333333]]),
    ]

    self._test_helper(
        minimization_problem,
        expected_states,
        regret_type="swap",
        update_type="additive")

  def test_multiplicative_swap_proxy_lagrangian_optimizer(self):
    """Tests that the stochastic matrices update as expected."""
    minimization_problem = test_util.ConstantMinimizationProblem(
        np.array([0.6, -0.1, 0.4]))
    # Calculated using a numpy+python implementation of the algorithm.
    expected_states = [
        np.log(
            np.array([[0.4, 0.4, 0.4, 0.4], [0.2, 0.2, 0.2, 0.2],
                      [0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]])),
        np.log(
            np.array([[0.36999014, 0.38528351, 0.38528351, 0.38528351],
                      [0.23517483, 0.21720297, 0.21720297, 0.21720297],
                      [0.17774131, 0.18882719, 0.18882719, 0.18882719],
                      [0.21709373, 0.20868632, 0.20868632, 0.20868632]])),
        np.log(
            np.array([[0.33972109, 0.36811863, 0.37118462, 0.36906575],
                      [0.27114826, 0.23738228, 0.23376693, 0.23626491],
                      [0.15712313, 0.17641793, 0.17858959, 0.17708679],
                      [0.23200752, 0.21808115, 0.21645886, 0.21758255]])),
    ]

    self._test_helper(
        minimization_problem,
        expected_states,
        regret_type="swap",
        update_type="multiplicative",
        initial_multiplier_radius=0.8)


if __name__ == "__main__":
  tf.test.main()
