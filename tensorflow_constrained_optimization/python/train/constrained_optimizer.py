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

from tensorflow_constrained_optimization.python import constrained_minimization_problem


# Change the base class from tf.train.Optimizer to tf.keras.optimizers.Optimizer
# to move to V2 optimizers. You will also need to:
#   1. Replace compute_gradients() with _compute_gradients()
#   2. Update the _create_slots(), _prepare(), _resource_apply_dense() and
#      _resourse_apply_sparse() implementations
#   3. Remove _apply_dense() and _apply_sparse()
#   4. Implement get_config() (it should probably just raise)
# The implementations of LagrangianOptimizer and ProxyLagrangianOptimizer should
# be unchanged. You *will* need to fix their tests, though.
@six.add_metaclass(abc.ABCMeta)
class ConstrainedOptimizer(tf.train.Optimizer):
  """Base class representing a constrained optimizer.

  A `ConstrainedOptimizer` wraps an `Optimizer` (or more than one), and applies
  it to a `ConstrainedMinimizationProblem`. Like an `Optimizer`, its minimize()
  method can be used to minimize a `Tensor` argument. Unlike a normal
  `Optimizer`, however, a `ConstrainedOptimizer` can *instead* take a
  `ConstrainedMinimizationProblem` as the first parameter to minimize(), in
  which case it will perform constrained optimization.

  A `ConstrainedOptimizer` wraps a normal `Optimizer` (the "optimizer"
  constructor parameter). If you minimize a `Tensor`, then the
  `ConstrainedOptimizer` will basically be an overly-complicated wrapper around
  this optimizer. The "constraint_optimizer" constructor parameter is used only
  for constrained optimization (i.e. when minimize() is given a
  `ConstrainedMinimizationProblem`), and is used to impose the constraints by
  maximizing over Lagrange multipliers (or their equivalents, if not using the
  Lagrangian formulation). If "constraint_optimizer" is not provided, then we
  will default to using the "optimizer" for the constraint portion of the
  optimization.
  """

  def __init__(self,
               optimizer,
               constraint_optimizer=None,
               name="ConstrainedOptimizer"):
    """Constructs a new `ConstrainedOptimizer`.

    Args:
      optimizer: `Optimizer`, used to optimize the objective and
        proxy_constraints portion of `ConstrainedMinimizationProblem`. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multipliers (or their analogues).
      constraint_optimizer: optional `Optimizer`, used to optimize the Lagrange
        multipliers (or their analogues).
      name: a non-empty string, which will be passed on to the parent
        `Optimizer`'s  constructor.
    """
    # I believe that use_locking does nothing here (we should just use whatever
    # locking behavior was given to the constructor(s) of the contained
    # optimizer(s)). The V2 optimizers don't take this parameter, so if the base
    # class of ConstrainedOptimizer is changed to tf.keras.optimizers.Optimizer,
    # then use_locking should be removed from the following __init__ call.
    super(ConstrainedOptimizer, self).__init__(use_locking=False, name=name)

    self._optimizer = optimizer
    self._constraint_optimizer = constraint_optimizer

  @property
  def optimizer(self):
    """Returns the `Optimizer` used for the objective and proxy_constraints."""
    return self._optimizer

  @property
  def constraint_optimizer(self):
    """Returns the `Optimizer` used for the Lagrange multiplier analogues."""
    return self._constraint_optimizer

  @abc.abstractmethod
  def _is_state(self, variable_or_handle):
    """Checks whether the given object is an internal state `Variable`.

    Must be overridden by all subclasses.

    `ConstrainedOptimizer` implementations contain internal state. For example,
    the `LagrangianOptimizer` contains a `Variable` of Lagrange multipliers.
    This function will return True iff the given "variable_or_handle"
    corresponds to such an internal state variable.

    The parameter is called "variable_or_handle" since, if it originates from
    the _apply_dense() or _apply_sparse() `Optimizer` methods, then it will be a
    variable, whereas if it originates from the _resource_apply_dense() or
    _resource_apply_sparse() methods, then it will be the handle of a resource
    variable.

    Args:
      variable_or_handle: the `Variable` to check.

    Returns:
      True iff variable_or_handle corresponds to an internal state variable of
      this `ConstrainedOptimizer`.
    """
    pass

  @abc.abstractmethod
  def _compute_constrained_minimization_gradients(self, compute_gradients_fn,
                                                  minimization_problem):
    """Computes gradients of a `ConstrainedMinimizationProblem`.

    Must be overridden by all subclasses.

    This is the analogue of the "compute_gradients" method of a normal
    `Optimizer`, *except* that, instead of differentiating a loss, it
    differentiates a `ConstrainedMinimizationProblem`. The compute_gradients_fn
    argument can be used by implementations to differentiate losses.

    For example, if the optimizer uses the Lagrangian formulation, then the
    implementation of this method would construct the Lagrangian for the given
    minimization_problem, and then call compute_gradients_fn to differentiate
    it.

    Args:
      compute_gradients_fn: a function taking a loss (a `Tensor` in graph mode,
        or a nullary function returning a `Tensor` in eager mode), and returning
        a list of (gradient, variable) pairs.
      minimization_problem: `ConstrainedMinimizationProblem`, the problem to
        differentiate. The actual function that will be differentiated depends
        on the formulation used by the `ConstrainedOptimizer` implementation
        (for example, the `LagrangianOptimizer` will differentiate the
        Lagrangian).
    """
    pass

  def compute_gradients(self,
                        minimization_problem,
                        var_list=None,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients of a `ConstrainedMinimizationProblem` (or loss).

    If minimization_problem is a `ConstrainedMinimizationProblem` (which is
    the most common use-case for this method), then there is one significant
    difference compated to the compute_gradients() method of a normal
    `Optimizer`: regardless of the value of the var_list parameter, the returned
    list of (gradient, variable) pairs will *always* include entries for all
    internal state variables of the `ConstrainedOptimizer` implementation (e.g.
    the Lagrange multipliers, for a `LagrangianOptimizer`).

    Inside apply_gradients(), these extra gradients will be dispatched to the
    appropriate contained optimizer (i.e. either the "optimizer" or
    "constrained_optimizer"), and optionally projected, in the
    {,_resource}_apply_{dense,sparse}() methods.

    If minimization_problem is *not* a `ConstrainedMinimizationProblem`, then
    this function will thunk down to the compute_gradients() method of the
    contained "optimizer".

    Args:
      minimization_problem: either a `ConstrainedMinimizationProblem`, or, if we
        do not wish to perform constrained optimization, a loss `Tensor` (in
        graph mode) or a nullary function returning a loss `Tensor` (in eager
        mode). In the two latter cases, this function will thunk down to the
        compute_gradients() method of the contained "optimizer".
      var_list: as in `tf.train.Optimizer`.
      gate_gradients: as in `tf.train.Optimizer`.
      aggregation_method: as in `tf.train.Optimizer`.
      colocate_gradients_with_ops: as in `tf.train.Optimizer`.
      grad_loss: as in `tf.train.Optimizer`.

    Returns:
      A list of (gradient, variable) pairs, as in the compute_gradients() mtehod
      of `tf.train.Optimizer`.
    """

    def compute_gradients_fn(loss):
      """Computes gradients of the given loss."""
      return super(ConstrainedOptimizer, self).compute_gradients(
          loss,
          var_list=var_list,
          gate_gradients=gate_gradients,
          aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops,
          grad_loss=grad_loss)

    if not isinstance(
        minimization_problem,
        constrained_minimization_problem.ConstrainedMinimizationProblem):
      return compute_gradients_fn(minimization_problem)

    if grad_loss is not None:
      raise ValueError("the grad_loss argument cannot be provided when the "
                       "loss argument is a ConstrainedMinimizationProblem")

    return self._compute_constrained_minimization_gradients(
        compute_gradients_fn, minimization_problem)

  # pylint: disable=protected-access

  def _create_slots(self, var_list):
    if self._constraint_optimizer is None:
      return self._optimizer._create_slots(var_list)

    optimizer_var_list = []
    constraint_optimizer_var_list = []
    for variable in var_list:
      is_state = self._is_state(variable)
      if is_state:
        constraint_optimizer_var_list.append(variable)
      else:
        optimizer_var_list.append(variable)
    self._optimizer._create_slots(optimizer_var_list)
    self._constraint_optimizer._create_slots(constraint_optimizer_var_list)

  def _prepare(self):
    self._optimizer._prepare()
    if self._constraint_optimizer is not None:
      self._constraint_optimizer._prepare()

  def _apply_dense(self, gradient, variable):
    if self._constraint_optimizer is not None and self._is_state(variable):
      return self._constraint_optimizer._apply_dense(gradient, variable)
    return self._optimizer._apply_dense(gradient, variable)

  def _apply_sparse(self, gradient, variable):
    if self._constraint_optimizer is not None and self._is_state(variable):
      return self._constraint_optimizer._apply_sparse(gradient, variable)
    return self._optimizer._apply_sparse(gradient, variable)

  def _resource_apply_dense(self, gradient, handle):
    if self._constraint_optimizer is not None and self._is_state(handle):
      return self._constraint_optimizer._resource_apply_dense(gradient, handle)
    return self._optimizer._resource_apply_dense(gradient, handle)

  def _resource_apply_sparse(self, gradient, handle):
    if self._constraint_optimizer is not None and self._is_state(handle):
      return self._constraint_optimizer._resource_apply_sparse(gradient, handle)
    return self._optimizer._resource_apply_sparse(gradient, handle)

  # pylint: enable=protected-access
