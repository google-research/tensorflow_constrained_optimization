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
               maximum_multiplier_radius=None):
    """Constructs a new `LagrangianOptimizer`.

    The difference between "optimizer" and "constraint_optimizer" (if the latter
    is provided) is that the former is used for learning the model parameters,
    while the latter us used for the Lagrange multipliers. If no
    "constraint_optimizer" is provided, then "optimizer" is used for both.

    Args:
      optimizer: `Optimizer`, used to optimize the objective and
        proxy_constraints portion of `ConstrainedMinimizationProblem`. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multipliers.
      constraint_optimizer: optional `Optimizer`, used to optimize the Lagrange
        multipliers.
      maximum_multiplier_radius: float, an optional upper bound to impose on the
        sum of the Lagrange multipliers.

    Raises:
      ValueError: if the maximum_multiplier_radius parameter is nonpositive.
    """
    super(LagrangianOptimizer, self).__init__(optimizer=optimizer)
    self._constraint_optimizer = constraint_optimizer

    if maximum_multiplier_radius and (maximum_multiplier_radius <= 0.0):
      raise ValueError("maximum_multiplier_radius must be strictly positive")

    self._maximum_multiplier_radius = maximum_multiplier_radius

    # We can't create the Lagrange multipliers here, since we don't know how
    # many constraints there will be until we see the
    # ConstrainedMinimizationProblem. Instead, we do so lazily, in
    # _maybe_create_multipliers().
    self._multipliers = None

  @property
  def constraint_optimizer(self):
    """Returns the `Optimizer` used for the Lagrange multipliers."""
    return self._constraint_optimizer

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
          name="lagrangian_state",
          dtype=tf.float32,
          use_resource=True)

    return self._multipliers

  def _projection_op(self, multipliers, name=None):
    """Projects the Lagrange multipliers onto the feasible region."""
    with tf.colocate_with(multipliers):
      if self._maximum_multiplier_radius:
        projected_multipliers = _project_multipliers_wrt_euclidean_norm(
            multipliers, self._maximum_multiplier_radius)
      else:
        projected_multipliers = tf.maximum(multipliers, 0.0)
      return multipliers.assign(projected_multipliers, name=name)

  def _minimize_constrained(self,
                            minimization_problem,
                            global_step=None,
                            var_list=None,
                            name=None):
    """Returns an `Operation` for minimizing the constrained problem.

    The "optimizer" constructor parameter will be used to update the model
    parameters, while the Lagrange multipliers will be updated using
    "constraint_optimizer" (if provided) or "optimizer" (if not).

    Args:
      minimization_problem: `ConstrainedMinimizationProblem`, the problem to
        optimize.
      global_step: as in `Optimizer`'s minimize() method.
      var_list: as in `Optimizer`'s minimize() method.
      name: as in `Optimizer`'s minimize() method.

    Returns:
      `Operation`, the train_op.

    Raises:
      RuntimeError: if you attempt to use this LagrangianOptimizer on two
        problems with different numbers of constraints.
      TypeError: if the "minimization_problem" `Tensor`s have different dtypes.
    """
    # Create the Lagrange multipliers.
    num_constraints = minimization_problem.num_constraints
    multipliers = self._maybe_create_multipliers(num_constraints)

    def loss_fn():
      """Evaluates the loss for the current Lagrange multipliers."""
      objective = minimization_problem.objective()
      proxy_constraints = minimization_problem.proxy_constraints()
      if proxy_constraints is None:
        proxy_constraints = minimization_problem.constraints()

      # Make sure that the objective and proxy constraints have the same dtype.
      if objective.dtype.base_dtype != proxy_constraints.dtype.base_dtype:
        raise TypeError("objective and proxy_constraints must have the same "
                        "dtypes")

      # Flatten the proxy constraint tensor to 1d.
      proxy_constraints = tf.reshape(
          proxy_constraints, shape=(num_constraints,))

      return (objective + tf.tensordot(
          tf.cast(multipliers, proxy_constraints.dtype), proxy_constraints, 1))

    # Create a list of (gradient,variable) pairs for the model parameters.
    grads_and_vars = self.optimizer.compute_gradients(loss_fn, var_list)

    # Compute the gradient w.r.t. the Lagrange multipliers (which is simply the
    # value of the constraint functions).
    constraints = minimization_problem.constraints()
    constraints = tf.reshape(constraints, shape=(num_constraints,))
    multipliers_gradient = tf.cast(constraints, multipliers.dtype)

    # Create a one-element list of (gradient,variable) pairs for the Lagrange
    # multipliers. We negate multipliers_gradient since we want to maximize
    # over the Lagrange multipliers.
    multiplier_grads_and_vars = [(-multipliers_gradient, multipliers)]

    update_ops = []
    if self._constraint_optimizer is None:
      # If we don't have a separate constraint_optimizer, then we use
      # self._optimizer for both the update of the model parameters, and that of
      # the Lagrange multipliers.
      update_ops.append(
          self.optimizer.apply_gradients(
              grads_and_vars + multiplier_grads_and_vars, name="update"))
    else:
      # If we have a separate constraint_optimizer, then we use self._optimizer
      # for the update of the model parameters, and self._constraint_optimizer
      # for that of the Lagrange multipliers.
      gradients = [
          gradient for gradient, _ in grads_and_vars + multiplier_grads_and_vars
          if gradient is not None
      ]
      # We need this tf.control_dependencies() block so that the Lagrange
      # multipliers don't get updated before the model parameter gradients are
      # computed.
      with tf.control_dependencies(gradients):
        update_ops.append(
            self.optimizer.apply_gradients(grads_and_vars, name="update"))
        update_ops.append(
            self._constraint_optimizer.apply_gradients(
                multiplier_grads_and_vars, name="optimizer_state_update"))

    with tf.control_dependencies(update_ops):
      if global_step is None:
        # If we don't have a global step, just project, and we're done.
        return self._projection_op(multipliers, name=name)
      else:
        # If we have a global step, then we need to increment it in addition to
        # projecting.
        projection_op = self._projection_op(
            multipliers, name="optimizer_state_project")
        with tf.colocate_with(global_step):
          global_step_op = global_step.assign_add(
              1, name="global_step_increment")
        return tf.group(projection_op, global_step_op, name=name)
