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
"""Tests for lagrangian_model_optimizer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.train import lagrangian_model_optimizer
from tensorflow_constrained_optimization.python.train import test_util
# Placeholder for internal import.


class Lagrangian(tf.Module):
  """Implements the usual Lagrangian formulation, as a Lagrangian model."""

  def __init__(self, num_constraints):
    super(Lagrangian, self).__init__()
    initial_lagrange_multipliers = np.zeros((num_constraints,),
                                            dtype=np.float32)
    self._lagrange_multipliers = tf.Variable(
        initial_lagrange_multipliers, trainable=True, dtype=tf.float32)

  def __call__(self):
    return self._lagrange_multipliers


lagrangian_model_matrix = np.array(
    [[0.11058229, 0.19960903, 0.33058789, 0.62870306],
     [0.9244308, 0.63366474, 0.56414986, 0.93732929],
     [0.2260223, 0.55009204, 0.47293151, 0.51393873]],
    dtype=np.float32)


class LagrangianModel(tf.Module):
  """Implements a linear (matrix * state) Lagrangian model."""

  def __init__(self, num_constraints):
    super(LagrangianModel, self).__init__()
    # We require exactly three constraints, and create four parameters. The
    # matrix created above is the transformation between them.
    assert num_constraints == 3
    initial_state = np.zeros((4,), dtype=np.float32)
    self._state = tf.Variable(initial_state, trainable=True, dtype=tf.float32)

  def __call__(self):
    return tf.tensordot(lagrangian_model_matrix, self._state, axes=(1, 0))


# @run_all_tests_in_graph_and_eager_modes
class LagrangianModelOptimizerTest(
    graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests the Lagrangian formulation and associated helper functions."""

  def _test_helper(self, minimization_problem, create_fn, expected_states):
    """Tests create_lagrangian_model_loss on the given minimization_problem."""
    loss_fn, update_ops_fn, trainable_variables = (
        lagrangian_model_optimizer.create_lagrangian_model_loss(
            minimization_problem, create_fn))
    optimizer = tf.keras.optimizers.SGD(1.0)
    var_list = list(
        minimization_problem.trainable_variables) + list(trainable_variables)

    if tf.executing_eagerly():
      train_op_fn = lambda: optimizer.minimize(loss_fn, var_list)
    else:
      # If we're in graph mode, then we need to create the train_op before the
      # session, so that we know which variables need to be initialized.
      train_op = optimizer.minimize(loss_fn, var_list)
      train_op_fn = lambda: train_op

    if len(trainable_variables) != 1:
      raise RuntimeError("_test_helper expects the Lagrangian model state to "
                         "only include 1 trainable variable, not {}".format(
                             len(trainable_variables)))
    state_variable = trainable_variables[0]

    # Perform a few steps of training, and record the states (which are just
    # Lagrange multipliers, in this case).
    states = []
    with self.wrapped_session() as session:
      while len(states) < len(expected_states):
        session.run_ops(update_ops_fn)
        states.append(session.run(state_variable))
        session.run_ops(train_op_fn)

    # Compare the observed sequence of Lagrange multipliers to the expected one.
    for expected, actual in zip(expected_states, states):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)

  def test_lagrangian_optimizer(self):
    """Tests that the Lagrangian model works with the usual Lagrangian."""
    minimization_problem = test_util.ConstantMinimizationProblem(
        np.array([0.6, -0.1, 0.4]))

    # The second element alternates between 0 and -0.1 since the
    # _LagrangianModelFormulation takes the absolute value of the Lagrange
    # multipliers before using them.
    expected_states = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.6, -0.1, 0.4]),
        np.array([1.2, 0.0, 0.8]),
        np.array([1.8, -0.1, 1.2]),
    ]

    self._test_helper(minimization_problem, Lagrangian, expected_states)

  def test_lagrangian_model_optimizer(self):
    """Tests that the Lagrangian model works with a linear Lagrangian."""
    gradient = np.array([0.6, -0.1, 0.4])
    minimization_problem = test_util.ConstantMinimizationProblem(gradient)

    state = np.zeros((4,), dtype=np.float32)
    expected_states = []
    for _ in xrange(3):
      expected_states.append(np.abs(state))
      state += np.tensordot(lagrangian_model_matrix, gradient, axes=(0, 0))
    expected_states.append(np.abs(state))

    self._test_helper(minimization_problem, LagrangianModel, expected_states)


if __name__ == "__main__":
  tf.test.main()
