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
"""Defines base class for `ConstrainedOptimizer`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class ConstrainedOptimizer(object):
  """Base class representing a constrained optimizer.

  A `ConstrainedOptimizer` wraps a `Optimizer` (or more than one), and applies
  it to a `ConstrainedMinimizationProblem`. Unlike a `Optimizer`, which takes a
  `Tensor` to minimize as a parameter to its minimize() method, a constrained
  optimizer instead takes a `ConstrainedMinimizationProblem`.
  """

  def __init__(self, optimizer):
    """Constructs a new `ConstrainedOptimizer`.

    Args:
      optimizer: `Optimizer`, used to optimize the
        `ConstraintedMinimizationProblem`.
    """
    self._optimizer = optimizer

  @property
  def optimizer(self):
    """Returns the `Optimizer` used for optimization."""
    return self._optimizer

  @abc.abstractmethod
  def _minimize_constrained(self,
                            minimization_problem,
                            global_step=None,
                            var_list=None,
                            name=None):
    """Returns an `Operation` for minimizing the constrained problem.

    Implementations of this method should ignore the "pre_train_ops" property of
    the "minimization_problem". The public minimize() method will take care of
    executing these before the returned train_op.

    Args:
      minimization_problem: `ConstrainedMinimizationProblem`, the problem to
        optimize.
      global_step: as in `Optimizer`'s minimize() method.
      var_list: as in `Optimizer`'s minimize() method.
      name: as in `Optimizer`'s minimize() method.

    Returns:
      `Operation`, the train_op.
    """

  def minimize_unconstrained(self,
                             minimization_problem,
                             global_step=None,
                             var_list=None,
                             name=None):
    """Returns an `Operation` for minimizing the unconstrained problem.

    This function *ignores* the "constraints" (and "proxy_constraints") portion
    of the minimization problem entirely, and only minimizes "objective".

    Args:
      minimization_problem: `ConstrainedMinimizationProblem`, the problem to
        optimize.
      global_step: as in `Optimizer`'s minimize() method.
      var_list: as in `Optimizer`'s minimize() method.
      name: as in `Optimizer`'s minimize() method.

    Returns:
      `Operation`, the train_op.
    """
    # Use tf.control_dependencies() to ensure that pre_train_ops execute before
    # the train_op.
    with tf.control_dependencies(minimization_problem.pre_train_ops()):
      return self.optimizer.minimize(
          minimization_problem.objective,
          global_step=global_step,
          var_list=var_list,
          name=name)

  def minimize(self,
               minimization_problem,
               unconstrained_steps=None,
               global_step=None,
               var_list=None,
               name=None):
    """Returns an `Operation` for minimizing the constrained problem.

    The first time this method is called, it will create the internal state
    (e.g. the Lagrange multipliers, for the LagrangianOptimizer). Because of
    this, in graph mode you must create your train_op by calling minimize()
    *before* initializing the variables in your session, so that the internal
    state will be properly initialized. In eager mode, variables are initialized
    when they're created, so there is no such issue.

    This method performs constrained optimization, while also including optional
    functionality that combines unconstrained with constrained optimization. If
    "global_step" < "unconstrained_steps", then this method will perform an
    unconstrained update, and if "global_step" >= "unconstrained_steps", then it
    will perform a constrained update. If "unconstrained_steps" isn't provided,
    then it will always perform constrained optimization.

    The reason for this functionality is that it may be best to initialize the
    constrained optimizer with an approximate optimum of the unconstrained
    problem.

    Args:
      minimization_problem: `ConstrainedMinimizationProblem`, the problem to
        optimize.
      unconstrained_steps: optional int, the number of steps for which we should
        perform unconstrained updates, before transitioning to constrained
        updates. If this parameter is not `None`, then global_step must be
        provided.
      global_step: as in `Optimizer`'s minimize() method.
      var_list: as in `Optimizer`'s minimize() method.
      name: as in `Optimizer`'s minimize() method.

    Returns:
      `Operation`, the train_op.

    Raises:
      ValueError: if unconstrained_steps is provided, but global_step is not.
    """

    def constrained_train_op_callback():
      """Returns an `Operation` for minimizing the constrained problem."""
      return self._minimize_constrained(
          minimization_problem,
          global_step=global_step,
          var_list=var_list,
          name=name)

    def unconstrained_train_op_callback():
      """Returns an `Operation` for minimizing the unconstrained problem."""
      return self.optimizer.minimize(
          minimization_problem.objective,
          global_step=global_step,
          var_list=var_list,
          name=name)

    # Use tf.control_dependencies() to ensure that pre_train_ops execute before
    # the train_op.
    with tf.control_dependencies(minimization_problem.pre_train_ops()):
      if unconstrained_steps is not None:
        if global_step is None:
          raise ValueError(
              "global_step cannot be None if unconstrained_steps is provided")
        unconstrained_steps_tensor = tf.convert_to_tensor(unconstrained_steps)
        dtype = unconstrained_steps_tensor.dtype
        return tf.cond(
            tf.cast(global_step, dtype) < unconstrained_steps_tensor,
            true_fn=unconstrained_train_op_callback,
            false_fn=constrained_train_op_callback)
      return constrained_train_op_callback()
