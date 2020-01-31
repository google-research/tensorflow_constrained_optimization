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
"""Constrained optimization using the Lagrangian formulation.

The code in this file uses the Lagrangian formulation to minimize a constrained
optimizion problem specified using the `ConstrainedMinimizationProblem`
interface. Compared with the proxy-Lagrangian formulation, the Lagrangian is the
"simpler" and "cheaper" option, although it has less theoretical backing.

At least in theory, external regret minimization of the Lagrangian formulation
could converge more slowly than swap regret minimization using the
proxy-Lagrangian formulation, assuming that the `ConstrainedMinimizationProblem`
we're optimizing uses "proxy_constraints".

For more specifics, please refer to:

> Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
> Constrained Optimization". ALT'19.
> [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

The Lagrangian formulation can be found in Definition 1, and is discussed in
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

  The feasible region is the set of all vectors with non-negative elements that
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
  multipliers_dims = multipliers.shape.dims
  if multipliers_dims is None:
    raise ValueError("multipliers must have a known rank")
  if len(multipliers_dims) != 1:
    raise ValueError("multipliers must be rank 1 (it is rank %d)" %
                     len(multipliers_dims))
  dimension = multipliers_dims[0].value
  if dimension is None:
    raise ValueError("multipliers must have a fully-known shape")

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


class _LagrangianFormulation(constrained_optimizer.Formulation):
  """Contains the internal state for the Lagrangian formulation.

  To optimize using the Lagrangian formulation, we jointly minimize over the
  model parameters, and maximize over Lagrange multipliers, with the latter
  maximization using additive updates and an algorithm that minimizes external
  regret.
  """

  def __init__(self, maximum_multiplier_radius=None, dual_scale=1.0):
    """Constructs a new `_LagrangianFormulation`.

    Args:
      maximum_multiplier_radius: float, an optional upper bound to impose on the
        sum of the Lagrange multipliers.
      dual_scale: optional float defaulting to 1, a multiplicative scaling
        factor applied to gradients w.r.t. the Lagrange multipliers.

    Raises:
      ValueError: if "maximum_multiplier_radius" or "dual_scale" are
        non-positive.
    """
    if (maximum_multiplier_radius is not None and
        maximum_multiplier_radius <= 0.0):
      raise ValueError("maximum_multiplier_radius must be strictly positive")
    self._maximum_multiplier_radius = maximum_multiplier_radius

    if dual_scale <= 0.0:
      raise ValueError("dual_scale must be strictly positive")
    self._dual_scale = dual_scale

    # We can't create the Lagrange multipliers here, since we don't know how
    # many constraints there will be until we see the
    # ConstrainedMinimizationProblem. Instead, we do so lazily, in
    # create_state().
    self._multipliers = None

  @property
  def state(self):
    return self._multipliers

  def _project_multipliers(self, multipliers):
    """Projects the Lagrange multipliers onto the feasible region."""
    if self._maximum_multiplier_radius:
      projected_multipliers = _project_multipliers_wrt_euclidean_norm(
          multipliers, self._maximum_multiplier_radius)
    else:
      projected_multipliers = tf.maximum(0.0, multipliers)
    return projected_multipliers

  def create_state(self, num_constraints):
    """Fetches (or creates, if necessary) the Lagrange multipliers."""
    # For the Lagrangian formulation, the internal state is simply a tensor of
    # Lagrange multipliers with shape (m,), where m is the number of
    # constraints.
    #
    # FUTURE WORK: make the dtype a parameter.
    if self._multipliers is not None:
      # We created the internal state below, and it should be impossible for it
      # to have an unknown or non-one-dimensional shape.
      multipliers_dims = self._multipliers.shape.dims
      assert multipliers_dims is not None
      assert len(multipliers_dims) == 1

      # You can use this optimizer on multiple different problems (sharing the
      # Lagrange multipliers between them), but only if they have the same
      # number of constraints, and therefore the same number of Lagrange
      # multipliers.
      if multipliers_dims[0].value != num_constraints:
        raise RuntimeError(
            "if you use the same Lagrangian optimizer on multiple problems, "
            "then they must have the same number of constraints, so that the "
            "Lagrange multipliers can be shared")

    else:
      initial_multipliers = np.zeros((num_constraints,), dtype=np.float32)
      self._multipliers = tf.compat.v2.Variable(
          initial_multipliers,
          trainable=True,
          name="tfco_lagrange_multipliers",
          dtype=tf.float32,
          constraint=self._project_multipliers)

    return self._multipliers

  def get_loss_fn(self, minimization_problem):
    """Returns the Lagrangian loss function.

    The resulting loss function uses `tf.custom_gradient` to override its
    gradients. In particular, the gradients w.r.t. the Lagrange multipliers will
    be negated (since we wish to maximize them), and will be written in terms of
    the constraints, instead of the proxy_constraints.

    Args:
      minimization_problem: `ConstrainedMinimizationProblem`, the problem to
        minimize.

    Returns:
      The Lagrangian loss function.
    """
    # This function returns both the value of the Lagrangian, and its gradients
    # w.r.t. the contents of the constrained minimization problem (objective,
    # constraints and proxy_constraints) and the internal Lagrange multipliers
    # (multipliers). The reason for using tf.custom_gradient is that it allows
    # us to override the gradient w.r.t. the Lagrange multipliers to use the
    # original constraints instead of the proxy constraints.
    @tf.custom_gradient
    def loss_gradient_fn(objective, constraints, proxy_constraints,
                         multipliers):
      """Evaluates the loss for the current Lagrange multipliers."""
      # Make sure that the objective and proxy constraints have the same dtype.
      if (constraints.dtype.base_dtype != objective.dtype.base_dtype or
          proxy_constraints.dtype.base_dtype != objective.dtype.base_dtype):
        raise TypeError("objective, constraints and proxy_constraints must all "
                        "have the same dtypes")

      output = objective + tf.tensordot(
          tf.cast(multipliers, dtype=proxy_constraints.dtype.base_dtype),
          proxy_constraints, 1)

      wrt_objective = 1
      wrt_proxy_constraints = tf.cast(
          multipliers, dtype=proxy_constraints.dtype.base_dtype)
      wrt_multipliers = -tf.cast(constraints, multipliers.dtype.base_dtype)

      def gradient_fn(output_gradient):
        # We return the gradient w.r.t. the objective, constraints,
        # proxy_constraints and Lagrange multipliers, respectively (this is the
        # same order as the arguments to loss_gradient_fn). Notice that the
        # gradient w.r.t. the constraints is None, and that w.r.t. the Lagrange
        # multipliers is scaled by dual_scale.
        return (output_gradient * wrt_objective, None,
                output_gradient * wrt_proxy_constraints,
                self._dual_scale * output_gradient * wrt_multipliers)

      return output, gradient_fn

    # Create the Lagrange multipliers.
    num_constraints = minimization_problem.num_constraints
    multipliers = self.create_state(num_constraints)

    # We don't use functools.partial since we need the arguments to be evaluated
    # when the loss is called, not when we construct the partial application.
    def partial_loss_gradient_fn():
      constraints = tf.reshape(
          minimization_problem.constraints(), shape=(num_constraints,))
      proxy_constraints = minimization_problem.proxy_constraints()
      if proxy_constraints is None:
        proxy_constraints = constraints
      else:
        proxy_constraints = tf.reshape(
            proxy_constraints, shape=(num_constraints,))
      return loss_gradient_fn(minimization_problem.objective(), constraints,
                              proxy_constraints, multipliers)

    return partial_loss_gradient_fn


def create_lagrangian_loss(minimization_problem,
                           maximum_multiplier_radius=None,
                           dual_scale=1.0):
  """Creates a loss function from a `ConstrainedMinimizationProblem`.

  Minimizing the returned loss will have the effect of jointly minimizing over
  the model parameters, and maximizing over Lagrange multipliers, with the
  latter maximization using additive updates and an algorithm that minimizes
  external regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used here--which is simply the usual Lagrangian
  formulation--can be found in Definition 1, and is discussed in Section 3. It
  is most similar to Algorithm 3 in Appendix C.3, with the two differences being
  that it uses proxy constraints (if they're provided) in the update of the
  model parameters, and uses `Optimizer`s, instead of SGD, for the "inner"
  updates.

  In addition to a loss function, this method returns a function returning a
  list of operations that should be executed before each iteration
  ("update_ops"), and a `tf.Variable` containing the Lagrange multipliers.

  In graph mode, the result of this update_ops function could be "attached" to
  the train_op using tf.control_dependencies. In eager mode, it should be called
  before each iteration. Likewise, you should make sure to differentiate w.r.t.
  the Lagrange multipliers variable by e.g. including it in the var_list that
  you pass to an `Optimizer`'s minimize() method.

  For example, in graph mode, your code could look like this:

  ```python
  # We ignore the returned Lagrange multipliers, since we won't provide a
  # var_list to the Optimizer's minimize() method.
  loss_fn, update_ops_fn, _ = create_loss(formulation, minimization_problem)

  optimizer = tf.compat.v1.train.GradientDescentOptimizer()
  with tf.control_dependencies(update_ops_fn()):
    # Since we don't provide a var_list, we'll just differentiate w.r.t. all
    # trainable variables, including the Lagrange multipliers.
    train_op = optimizer.minimize(loss_fn())

  with tf.compat.v1.Session() as session:
    for iteration in xrange(num_iterations):
      session.run(train_op)
  ```

  while in eager mode, it could look like this:

  ```python
  loss_fn, update_ops_fn, multipliers_variable = create_loss(formulation,
      minimization_problem)

  # Assuming that we already have a var_list containing the model parameters.
  var_list += minimization_problem.trainable_variables
  var_list.append(multipliers_variable)

  optimizer = tf.keras.optimizers.SGD()

  for iteration in xrange(num_iterations):
    update_ops_fn()
    optimizer.minimize(loss_fn, var_list=var_list)
  ```

  Args:
    minimization_problem: `ConstrainedMinimizationProblem`, the problem to
      optimize.
    maximum_multiplier_radius: float, an optional upper bound to impose on the
      sum of the Lagrange multipliers.
    dual_scale: optional float defaulting to 1, a multiplicative scaling factor
      applied to gradients w.r.t. the Lagrange multipliers.

  Returns:
    A (loss_fn, update_ops_fn, multipliers_variable) tuple, where loss_fn is a
    nullary function returning a `Tensor` that can be minimized to optimize the
    constrained problem, update_ops_fn is a nullary function that returns a list
    of operations that should be executed before each training iteration, and
    multipliers_variable is a `tf.Variable` of Lagrange multipliers.
  """
  return constrained_optimizer.create_loss(
      _LagrangianFormulation(
          maximum_multiplier_radius=maximum_multiplier_radius,
          dual_scale=dual_scale), minimization_problem)


class LagrangianOptimizerV1(constrained_optimizer.ConstrainedOptimizerV1):
  """A `ConstrainedOptimizerV1` based on the Lagrangian formulation.

  This constrained optimizer uses the given `tf.compat.v1.train.Optimizer`s to
  jointly minimize over the model parameters, and maximize over Lagrange
  multipliers, with the latter maximization using additive updates and an
  algorithm that minimizes external regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer--which is simply the usual Lagrangian
  formulation--can be found in Definition 1, and is discussed in Section 3. It
  is most similar to Algorithm 3 in Appendix C.3, with the two differences being
  that it uses proxy constraints (if they're provided) in the update of the
  model parameters, and uses `tf.compat.v1.train.Optimizer`s, instead of SGD,
  for the "inner" updates.

  The Lagrange multipliers are owned by this optimizer. Hence, if you want to
  use a `LagrangianOptimizerV1` on multiple `ConstrainedMinimizationProblem`s,
  while sharing the Lagrange multipliers between them, then you may do so.
  However, each problem must have the same number of constraints (an exception
  will be raised, otherwise), so that the Lagrange multipliers are compatible.
  """

  def __init__(self,
               optimizer,
               num_constraints=None,
               constraint_optimizer=None,
               maximum_multiplier_radius=None,
               name="LagrangianOptimizerV1"):
    """Constructs a new `LagrangianOptimizerV1`.

    The difference between "optimizer" and "constraint_optimizer" (if the latter
    is provided) is that the former is used for learning the model parameters,
    while the latter us used for the Lagrange multipliers. If no
    "constraint_optimizer" is provided, then "optimizer" is used for both.

    Args:
      optimizer: `tf.compat.v1.train.Optimizer`, used to optimize the objective
        and proxy_constraints portion of `ConstrainedMinimizationProblem`. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multipliers.
      num_constraints: optional int, the number of constraints in the
        `ConstrainedMinimizationProblem` that will eventually be minimized. If
        this argument is provided, then the Lagrange multipliers will be created
        inside this constructor. Otherwise, they'll be created inside the first
        call to get_loss_fn().
      constraint_optimizer: optional `tf.compat.v1.train.Optimizer`, used to
        optimize the Lagrange multipliers.
      maximum_multiplier_radius: float, an optional upper bound to impose on the
        sum of the Lagrange multipliers.
      name: as in `ConstrainedOptimizerV1`.

    Raises:
      ValueError: if the maximum_multiplier_radius parameter is non-positive.
    """
    super(LagrangianOptimizerV1, self).__init__(
        _LagrangianFormulation(
            maximum_multiplier_radius=maximum_multiplier_radius),
        optimizer=optimizer,
        num_constraints=num_constraints,
        constraint_optimizer=constraint_optimizer,
        name=name)


class LagrangianOptimizerV2(constrained_optimizer.ConstrainedOptimizerV2):
  """A `ConstrainedOptimizerV2` based on the Lagrangian formulation.

  This constrained optimizer uses the given `tf.keras.optimizers.Optimizer`s to
  jointly minimize over the model parameters, and maximize over Lagrange
  multipliers, with the latter maximization using additive updates and an
  algorithm that minimizes external regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer--which is simply the usual Lagrangian
  formulation--can be found in Definition 1, and is discussed in Section 3. It
  is most similar to Algorithm 3 in Appendix C.3, with the two differences being
  that it uses proxy constraints (if they're provided) in the update of the
  model parameters, and uses `tf.keras.optimizers.Optimizer`s, instead of SGD,
  for the "inner" updates.

  The Lagrange multipliers are owned by this optimizer. Hence, if you want to
  use a `LagrangianOptimizerV2` on multiple `ConstrainedMinimizationProblem`s,
  while sharing the Lagrange multipliers between them, then you may do so.
  However, each problem must have the same number of constraints (an exception
  will be raised, otherwise), so that the Lagrange multipliers are compatible.
  """

  def __init__(self,
               optimizer,
               num_constraints,
               constraint_optimizer=None,
               maximum_multiplier_radius=None,
               name="LagrangianOptimizerV2"):
    """Constructs a new `LagrangianOptimizerV2`.

    The difference between "optimizer" and "constraint_optimizer" (if the latter
    is provided) is that the former is used for learning the model parameters,
    while the latter us used for the Lagrange multipliers. If no
    "constraint_optimizer" is provided, then "optimizer" is used for both.

    Args:
      optimizer: `tf.keras.optimizers.Optimizer`, used to optimize the objective
        and proxy_constraints portion of `ConstrainedMinimizationProblem`. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multipliers.
      num_constraints: int, the number of constraints in the
        `ConstrainedMinimizationProblem` that will eventually be minimized.
      constraint_optimizer: optional `tf.keras.optimizers.Optimizer`, used to
        optimize the Lagrange multipliers.
      maximum_multiplier_radius: float, an optional upper bound to impose on the
        sum of the Lagrange multipliers.
      name: as in `ConstrainedOptimizerV2`.

    Raises:
      ValueError: if the maximum_multiplier_radius parameter is non-positive.
    """
    super(LagrangianOptimizerV2, self).__init__(
        _LagrangianFormulation(
            maximum_multiplier_radius=maximum_multiplier_radius),
        optimizer=optimizer,
        num_constraints=num_constraints,
        constraint_optimizer=constraint_optimizer,
        name=name)
