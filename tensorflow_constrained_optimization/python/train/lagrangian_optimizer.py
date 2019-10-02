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
"""Defines `LagrangianOptimizer`.

A `LagrangianOptimizer` is used to minimize a constrained optimizion problem
specified using the `ConstrainedMinimizationProblem` interface. Compared with
the `ProxyLagrangianOptimizer`, `LagrangianOptimizer` is the "simpler" and
"cheaper" option, although it has less theoretical backing.

Specifically, a `LagrangianOptimizer` introduces Lagrange multipliers, and uses
contained `Optimizer`s to jointly optimize over the model parameters and
Lagrange multipliers.

At least in theory, external regret minimization of the Lagrangian formulation
(i.e. using the `LagrangianOptimizer` defined here) suffices if the
`ConstrainedMinimizationProblem` we're optimizing doesn't have any
"proxy_constraints", while swap regret minimization of the proxy-Lagrangian
formulation (using the `ProxyLagrangianOptimizer`) is recommended if
"proxy_constraints" are present. In practice, the `LagrangianOptimizer` works in
the latter case, but seemingly not quite as well as the
`ProxyLagrangianOptimizer`.

For more specifics, please refer to:

> Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
> Constrained Optimization". ALT'19.
> [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

The formulation used by the LagrangianOptimizer--which is simply the usual
Lagrangian formulation--can be found in Definition 1, and is discussed in
Section 3. This optimizer is most similar to Algorithm 3 in Appendix C.3, with
the two differences being that it uses proxy constraints (if they're provided)
in the update of the model parameters, and uses `Optimizer`s, instead of SGD,
for the "inner" updates.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_constrained_optimization.python.train import constrained_optimizer


def _project_multipliers_wrt_euclidean_norm(multipliers, radius):
  """Projects its argument onto the feasible region.

  The feasible region is the set of all vectors with nonnegative elements that
  sum to at most "radius".

  Args:
    multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
    radius: float, the radius of the feasible region.

  Returns:
    The rank-1 `Tensor` that results from projecting "multipliers" onto the
    feasible region w.r.t. the Euclidean norm.

  Raises:
    TypeError: if the "multipliers" `Tensor` is not floating-point.
    ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
      or is not one-dimensional.
  """
  if not multipliers.dtype.is_floating:
    raise TypeError("multipliers must have a floating-point dtype")
  multipliers_shape = multipliers.shape.dims
  if multipliers_shape is None:
    raise ValueError("multipliers must have known shape")
  if len(multipliers_shape) != 1:
    raise ValueError(
        "multipliers must be one dimensional (instead is %d-dimensional)" %
        len(multipliers_shape))
  dimension = multipliers_shape[0].value
  if dimension is None:
    raise ValueError("multipliers must have fully-known shape")

  def while_loop_condition(iteration, multipliers, inactive, old_inactive):
    """Returns false if the while loop should terminate."""
    del multipliers  # Needed by the body, but not the condition.
    not_done = (iteration < dimension)
    not_converged = tf.reduce_any(tf.not_equal(inactive, old_inactive))
    return tf.logical_and(not_done, not_converged)

  def while_loop_body(iteration, multipliers, inactive, old_inactive):
    """Performs one iteration of the projection."""
    del old_inactive  # Needed by the condition, but not the body.
    iteration += 1
    scale = tf.minimum(0.0, (radius - tf.reduce_sum(multipliers)) /
                       tf.maximum(1.0, tf.reduce_sum(inactive)))
    multipliers = multipliers + (scale * inactive)
    new_inactive = tf.cast(multipliers > 0, multipliers.dtype)
    multipliers = multipliers * new_inactive
    return (iteration, multipliers, new_inactive, inactive)

  iteration = tf.constant(0)
  inactive = tf.ones_like(multipliers, dtype=multipliers.dtype)

  # We actually want a do-while loop, so we explicitly call while_loop_body()
  # once before tf.while_loop().
  iteration, multipliers, inactive, old_inactive = while_loop_body(
      iteration, multipliers, inactive, inactive)
  iteration, multipliers, inactive, old_inactive = tf.while_loop(
      while_loop_condition,
      while_loop_body,
      loop_vars=(iteration, multipliers, inactive, old_inactive),
      name="euclidean_projection")

  return multipliers


class LagrangianOptimizer(constrained_optimizer.ConstrainedOptimizer):
  """A `ConstrainedOptimizer` based on the Lagrangian formulation.

  This `ConstrainedOptimizer` uses the given `Optimizer`s to jointly minimize
  over the model parameters, and maximize over Lagrange multipliers, with the
  latter maximization using additive updates and an algorithm that minimizes
  external regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer--which is simply the usual Lagrangian
  formulation--can be found in Definition 1, and is discussed in Section 3. It
  is most similar to Algorithm 3 in Appendix C.3, with the two differences being
  that it uses proxy constraints (if they're provided) in the update of the
  model parameters, and uses `Optimizer`s, instead of SGD, for the "inner"
  updates.

  The Lagrange multipliers are owned by the optimizer. Hence, if you want to use
  a LagrangianOptimizer on multiple `ConstrainedMinimizationProblem`s, while
  sharing the Lagrange multipliers between them, then you may do so. However,
  each problem must have the same number of constraints (an exception will be
  raised, otherwise), so that the Lagrange multipliers are compatible.
  """

  def __init__(self,
               optimizer,
               constraint_optimizer=None,
               maximum_multiplier_radius=None,
               name="LagrangianOptimizer"):
    """Constructs a new `LagrangianOptimizer`.

    The difference between "optimizer" and "constraint_optimizer" (if the latter
    is provided) is that the former is used for learning the model parameters,
    while the latter us used for the Lagrange multipliers. If no
    "constraint_optimizer" is provided, then "optimizer" is used for both.

    Args:
      optimizer: as in `ConstrainedOptimizer`.
      constraint_optimizer: as in `ConstrainedOptimizer`.
      maximum_multiplier_radius: float, an optional upper bound to impose on the
        sum of the Lagrange multipliers.
      name: as in `ConstrainedOptimizer`.

    Raises:
      ValueError: if the maximum_multiplier_radius parameter is nonpositive.
    """
    super(LagrangianOptimizer, self).__init__(
        optimizer=optimizer,
        constraint_optimizer=constraint_optimizer,
        name=name)

    if maximum_multiplier_radius and (maximum_multiplier_radius <= 0.0):
      raise ValueError("maximum_multiplier_radius must be strictly positive")

    self._maximum_multiplier_radius = maximum_multiplier_radius

    # We can't create the Lagrange multipliers here, since we don't know how
    # many constraints there will be until we see the
    # ConstrainedMinimizationProblem. Instead, we do so lazily, in
    # _maybe_create_multipliers().
    self._multipliers = None

  def _project_multipliers(self, multipliers):
    """Projects the Lagrange multipliers onto the feasible region."""
    if self._maximum_multiplier_radius:
      projected_multipliers = _project_multipliers_wrt_euclidean_norm(
          multipliers, self._maximum_multiplier_radius)
    else:
      projected_multipliers = tf.maximum(0.0, multipliers)
    return projected_multipliers

  def _maybe_create_multipliers(self, num_constraints):
    """Fetches/creates the internal state."""
    # For a LagrangianOptimizer, the internal state is simply a tensor of
    # Lagrange multipliers with shape (m,), where m is the number of
    # constraints.
    #
    # FUTURE WORK: make the dtype a parameter.
    if self._multipliers is not None:
      dims = self._multipliers.shape.dims

      # We created the internal state below, and it should be impossible for it
      # to have an unknown or non-one-dimensional shape.
      assert dims is not None
      assert len(dims) == 1

      # You can use this optimizer on multiple different problems (sharing the
      # Lagrange multipliers between them), but only if they have the same
      # number of constraints, and therefore the same number of Lagrange
      # multipliers.
      if dims[0].value != num_constraints:
        raise RuntimeError(
            "if you use the same LagrangianOptimizer on multiple problems, "
            "then they must have the same number of constraints, so that the "
            "Lagrange multipliers can be shared")

    else:
      initial_multipliers = np.zeros((num_constraints,), dtype=np.float32)
      self._multipliers = tf.Variable(
          initial_multipliers,
          trainable=False,
          name="lagrange_multipliers",
          dtype=tf.float32,
          constraint=self._project_multipliers,
          use_resource=True)

    return self._multipliers

  def _is_state(self, variable_or_handle):
    # This assertion should always succeed, since the Lagrange multipliers will
    # be created inside compute_gradients(), whereas this function will only be
    # called from within apply_gradients().
    assert self._multipliers is not None

    # This comparison seems to work even if self._state is a resource variable,
    # and variable_or_handle is a handle coming from _resource_apply_dense() or
    # _resource_apply_sparse(). This might not be strictly correct, though.
    return variable_or_handle == self._multipliers

  def _compute_constrained_minimization_gradients(self, compute_gradients_fn,
                                                  minimization_problem):
    # Create the Lagrange multipliers.
    num_constraints = minimization_problem.num_constraints
    multipliers = self._maybe_create_multipliers(num_constraints)

    # The "primal" portion of the Lagrangian game.
    def loss_fn():
      """Evaluates the loss for the current Lagrange multipliers."""
      objective = minimization_problem.objective()
      proxy_constraints = minimization_problem.proxy_constraints()
      if proxy_constraints is None:
        proxy_constraints = minimization_problem.constraints()

      # Make sure that the objective and proxy constraints have the same
      # dtype.
      if objective.dtype.base_dtype != proxy_constraints.dtype.base_dtype:
        raise TypeError("objective and proxy_constraints must have the same "
                        "dtypes")

      # Flatten the proxy constraint tensor to 1d.
      proxy_constraints = tf.reshape(
          proxy_constraints, shape=(num_constraints,))

      return (objective + tf.tensordot(
          tf.cast(multipliers, proxy_constraints.dtype), proxy_constraints, 1))

    with tf.control_dependencies(minimization_problem.pre_train_ops()):
      # Create a list of (gradient,variable) pairs for the model parameters. In
      # graph mode, compute_gradients() expects a Tensor, while in eager mode,
      # it expects a function returning a Tensor.
      if not tf.executing_eagerly():
        loss_fn = loss_fn()
      grads_and_vars = compute_gradients_fn(loss_fn)

      # Compute the gradient w.r.t. the Lagrange multipliers (the "dual" portion
      # of the Lagrangian game. This gradient is simply the value of the
      # constraint functions).
      constraints = minimization_problem.constraints()
      constraints = tf.reshape(constraints, shape=(num_constraints,))
      multipliers_gradient = tf.cast(constraints, multipliers.dtype)

      # Create a (gradient,variable) pair for the Lagrange multipliers. We
      # negate multipliers_gradient since we want to maximize over the Lagrange
      # multipliers.
      grads_and_vars.append((-multipliers_gradient, multipliers))

      return grads_and_vars


def create_lagrangian_loss(minimization_problem,
                           maximum_multiplier_radius=None):
  """Creates a loss function from a `ConstrainedMinimizationProblem`.

  In addition to a loss function, this method returns a function returning a
  list of operations that should be executed before each iteration ("pre-train
  ops").

  In graph mode, the result of this pre-train ops function could be "attached"
  to the train_op using tf.control_dependencies. In eager mode, it should be
  called before each iteration.

  Args:
    minimization_problem: `ConstrainedMinimizationProblem`, the problem to
      optimize.
    maximum_multiplier_radius: float, an optional upper bound to impose on the
      sum of the Lagrange multipliers.

  Returns:
    A (loss_fn, pre_train_ops_fn, lagrange_multipliers) tuple, where loss_fn is
    a nullary function returning a `Tensor` that can be minimized to optimize
    the Lagrangian, pre_train_ops_fn is a nullary function that returns a list
    of operations that should be executed before each training iteration, and
    lagrange_multipliers is a tf.Variable of Lagrange multipliers.
  """

  def constraint_fn(multipliers):
    """Projects the Lagrange multipliers onto the feasible region."""
    if maximum_multiplier_radius:
      projected_multipliers = _project_multipliers_wrt_euclidean_norm(
          multipliers, maximum_multiplier_radius)
    else:
      projected_multipliers = tf.maximum(0.0, multipliers)
    return projected_multipliers

  initial_multipliers = np.zeros((minimization_problem.num_constraints,),
                                 dtype=np.float32)
  multipliers = tf.Variable(
      initial_multipliers,
      trainable=True,
      name="lagrange_multipliers",
      dtype=tf.float32,
      constraint=constraint_fn,
      use_resource=True)

  def loss_fn():
    """Returns a `Tensor` that should be minimized."""
    objective = minimization_problem.objective()
    constraints = minimization_problem.constraints()
    proxy_constraints = minimization_problem.proxy_constraints()
    if proxy_constraints is None:
      proxy_constraints = constraints

    # We want to minimize the "primal" Tensor, and maximize the "dual" Tensor,
    # so we subtract them.
    primal = (
        objective + tf.tensordot(
            tf.stop_gradient(
                tf.cast(multipliers, proxy_constraints.dtype.base_dtype)),
            proxy_constraints, 1))
    dual = tf.tensordot(
        tf.cast(multipliers, constraints.dtype.base_dtype),
        tf.stop_gradient(constraints), 1)
    return primal - dual

  return loss_fn, minimization_problem.pre_train_ops, multipliers
