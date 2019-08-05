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
"""Defines `ProxyLagrangianOptimizer`.

A `ProxyLagrangianOptimizer` is used to minimize a constrained optimizion
problem specified using the `ConstrainedMinimizationProblem` interface.
Compared with the `LagrangianOptimizer`, `ProxyLagrangianOptimizer` is the more
complex and expensive option, although it has more theoretical backing.
Generally, the extra computational cost of using a `ProxyLagrangianOptimizer`
will be small, assuming that the minibatch size is sufficiently large compared
to the number of constraints. In this case, we recommend it over the
`LagrangianOptimizer`.

Specifically, a `ProxyLagrangianOptimizer` learns what weights should be
associated with the objective function and constraints using the so-called
"proxy-Lagrangian" formulation, which does *not* use Lagrange multipliers (but
the idea is similar). The main differences between the formulation used here,
and the standard Lagrangian formulation, are that (i) the objective function is
weighted, in addition to the constraints, and (ii) these "multipliers" are taken
from the (m+1)-dimensional simplex (m being the number of constraints), instead
of the m-dimensional positive orthant.

At least in theory, external regret minimization of the Lagrangian formulation
(using the `LagrangianOptimizer`) suffices if the
`ConstrainedMinimizationProblem` we're optimizing doesn't have any
"proxy_constraints", while swap regret minimization of the proxy-Lagrangian
formulation (using the `ProxyLagrangianOptimizer` defined here) is recommended
if "proxy_constraints" are present. In practice, the `LagrangianOptimizer` works
in the latter case, but seemingly not quite as well as the
`ProxyLagrangianOptimizer`.

For more specifics, please refer to:

> Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
> Constrained Optimization". ALT'19.
> [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

The formulation used by the `ProxyLagrangianOptimizer` is the so-called
"proxy-Lagrangian" formulation, which can be found in Definition 2, and is
discussed in Section 4. If using multiplicative updates to minimize swap regret
(the update_type and regret_type constructor parameters), the algorithm is most
similar to Algorithm 2 in Section 4, with the difference being that it uses
`Optimizer`s, instead of SGD, for the "inner" updates. Instead of swap regret,
one can alternatively use external regret, and instead of multiplicative
updates, one can alternatively use additive updates, if desired.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf

from tensorflow_constrained_optimization.python.train import constrained_optimizer

# The possible values of the regret_type constructor parameter.
_EXTERNAL_REGRET_TYPE = "external"
_SWAP_REGRET_TYPE = "swap"
# The possible values of the update_type constructor parameter.
_ADDITIVE_UPDATE_TYPE = "additive"
_MULTPILICATIVE_UPDATE_TYPE = "multiplicative"


def _maximal_eigenvector_power_method(matrix,
                                      epsilon=1e-6,
                                      maximum_iterations=100):
  """Returns a maximal right-eigenvector of "matrix" using the power method.

  Args:
    matrix: 2D Tensor, the matrix of which we will find a maximal
      right-eigenvector.
    epsilon: nonnegative float, if two iterations of the power method differ (in
      L2 norm) by no more than epsilon, we will terminate.
    maximum_iterations: nonnegative int, if we perform this many iterations, we
      will terminate.
  Result: A maximal right-eigenvector of "matrix".

  Raises:
    TypeError: if the "matrix" `Tensor` is not floating-point.
    ValueError: if the "epsilon" or "maximum_iterations" parameters violate
      their bounds.
  """
  if not matrix.dtype.is_floating:
    raise TypeError("multipliers must have a floating-point dtype")
  if epsilon <= 0.0:
    raise ValueError("epsilon must be strictly positive")
  if maximum_iterations <= 0:
    raise ValueError("maximum_iterations must be strictly positive")

  def while_loop_condition(iteration, eigenvector, old_eigenvector):
    """Returns false if the while loop should terminate."""
    not_done = (iteration < maximum_iterations)
    not_converged = (tf.norm(eigenvector - old_eigenvector) > epsilon)
    return tf.logical_and(not_done, not_converged)

  def while_loop_body(iteration, eigenvector, old_eigenvector):
    """Performs one iteration of the power method."""
    del old_eigenvector  # Needed by the condition, but not the body (for lint).
    iteration += 1
    # We need to use tf.matmul() and tf.expand_dims(), instead of
    # tf.tensordot(), since the former will infer the shape of the result, while
    # the latter will not (tf.while_loop() needs the shapes).
    new_eigenvector = tf.matmul(matrix, tf.expand_dims(eigenvector, 1))[:, 0]
    new_eigenvector /= tf.norm(new_eigenvector)
    return (iteration, new_eigenvector, eigenvector)

  iteration = tf.constant(0)
  eigenvector = tf.ones_like(matrix[:, 0])
  eigenvector /= tf.norm(eigenvector)

  # We actually want a do-while loop, so we explicitly call while_loop_body()
  # once before tf.while_loop().
  iteration, eigenvector, old_eigenvector = while_loop_body(
      iteration, eigenvector, eigenvector)
  iteration, eigenvector, old_eigenvector = tf.while_loop(
      while_loop_condition,
      while_loop_body,
      loop_vars=(iteration, eigenvector, old_eigenvector),
      name="power_method")

  return eigenvector


def _project_distribution_wrt_euclidean_norm(distribution):
  """Projects onto the set of multinoulli distributions.

  The projection is performed along the first axis. If the `Tensor` is of shape
  (n_0,n_1,n_2,...,n_d), then this algorithm is O(n_0^2 * n_1 * n_2 * ... *
  n_d) at most. It could be done in O(n_0 * log(n_0) * n_1 * n_2 * ... * n_d)
  time using a sort, (and maybe better with a different algorithm), but the
  algorithm implemented here is easier to implement in TensorFlow.

  Args:
    distribution: a `Tensor` to project onto the space of multinoulli
      distributions along its first axis.

  Returns:
    The result of projecting "distribution" onto the space of multinoulli
    distributions along its first axis, with respect to the Euclidean norm.

  Raises:
    TypeError: if the "distribution" `Tensor` is not floating-point.
    ValueError: if the "distribution" `Tensor` does not have a known size in its
      first dimension.
  """
  if not distribution.dtype.is_floating:
    raise TypeError("distribution must have a floating-point dtype")
  distribution_shape = distribution.shape.dims
  if distribution_shape is None:
    raise ValueError("distribution must have known shape")
  dimension = distribution_shape[0].value
  if dimension is None:
    raise ValueError("distribution must have fully-known shape")

  def while_loop_condition(iteration, distribution, inactive, old_inactive):
    """Returns false if the while loop should terminate."""
    del distribution  # Needed by the body, but not the condition (for lint).
    not_done = (iteration < dimension)
    not_converged = tf.reduce_any(tf.not_equal(inactive, old_inactive))
    return tf.logical_and(not_done, not_converged)

  def while_loop_body(iteration, distribution, inactive, old_inactive):
    """Performs one iteration of the projection."""
    del old_inactive  # Needed by the condition, but not the body (for lint).
    iteration += 1
    scale = (1.0 -
             tf.reduce_sum(distribution, axis=0, keepdims=True)) / tf.maximum(
                 1.0, tf.reduce_sum(inactive, axis=0, keepdims=True))
    distribution = distribution + (scale * inactive)
    new_inactive = tf.cast(distribution > 0, distribution.dtype)
    distribution = distribution * new_inactive
    return (iteration, distribution, new_inactive, inactive)

  iteration = tf.constant(0)
  inactive = tf.ones_like(distribution, dtype=distribution.dtype)

  # We actually want a do-while loop, so we explicitly call while_loop_body()
  # once before tf.while_loop().
  iteration, distribution, inactive, old_inactive = while_loop_body(
      iteration, distribution, inactive, inactive)
  iteration, distribution, inactive, old_inactive = tf.while_loop(
      while_loop_condition,
      while_loop_body,
      loop_vars=(iteration, distribution, inactive, old_inactive),
      name="euclidean_projection")

  return distribution


def _project_log_distribution_wrt_kl_divergence(log_distribution):
  """Projects onto the set of log-multinoulli distributions.

  Let "distribution" be the quantity that results from exponentiating
  "log_distribution" element-wise. The result of this function will then be the
  element-wise logarithm of new_distribution, which is chosen to minimize
  D_KL(new_distribution | distribution).

  Args:
    log_distribution: a `Tensor` to project onto the space of log-multinoulli
      distributions along its first axis.

  Returns:
    The element-wise logarithm of the result of projecting exp(log_distribution)
    onto the space of multinoulli distributions along its first axis, with
    respect to the KL-divergence
  """

  # For numerical reasons, make sure that the largest element is zero before
  # exponentiating.
  log_distribution = log_distribution - tf.reduce_max(
      log_distribution, axis=0, keepdims=True)
  log_distribution = log_distribution - tf.log(
      tf.reduce_sum(tf.exp(log_distribution), axis=0, keepdims=True))
  return log_distribution


class ProxyLagrangianOptimizer(constrained_optimizer.ConstrainedOptimizer):
  """A `ConstrainedOptimizer` based on the proxy-Lagrangian formulation.

  This `ConstrainedOptimizer` uses the given `Optimizer`s to jointly minimize
  over the model parameters, and maximize over constraint/objective weights (the
  analogue of Lagrange multipliers).

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer is the so-called "proxy-Lagrangian"
  formulation, which can be found in Definition 2, and is discussed in Section
  4. If using multiplicative updates to minimize swap regret (the update_type
  and regret_type constructor parameters), the algorithm is most similar to
  Algorithm 2 in Section 4, with the difference being that it uses
  `Optimizer`s, instead of SGD, for the "inner" updates. Instead of swap regret,
  one can alternatively use external regret, and instead of multiplicative
  updates, one can alternatively use additive updates, if desired.

  The internal state (the analogues of the Lagrange multipliers) are owned by
  the optimizer. Hence, if you want to use a ProxyLagrangianOptimizer on
  multiple `ConstrainedMinimizationProblem`s, while sharing this internal state
  between them, then you may do so. However, each problem must have the same
  number of constraints (an exception will be raised, otherwise), so that the
  internal states are compatible.
  """

  def __init__(self,
               optimizer,
               constraint_optimizer=None,
               regret_type=_SWAP_REGRET_TYPE,
               update_type=_MULTPILICATIVE_UPDATE_TYPE,
               minimum_multiplier_radius=None,
               initial_multiplier_radius=None):
    """Constructs a new `ProxyLagrangianOptimizer`.

    The difference between "optimizer" and "constraint_optimizer" (if the latter
    is provided) is that the former is used for learning the model parameters,
    while the latter us used for the update to the constraint/objective weights
    (the analogue of Lagrange multipliers). If no "constraint_optimizer" is
    provided, then "optimizer" is used for both.

    Args:
      optimizer: `Optimizer`, used to optimize the objective and
        proxy_constraints portion of `ConstrainedMinimizationProblem`. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multiplier analogues.
      constraint_optimizer: optional `Optimizer`, used to optimize the Lagrange
        multiplier analogues.
      regret_type: string, either "external" or "swap", determines what type of
        regret minimization to perform when updating the constraint/objective
        weights (the analogue of the Lagrange multipliers). This parameter has
        *no effect* on the optimization of the model parameters. Defaults to
        "swap".
      update_type: string, either "additive" or "multiplicative", determines
        what type of updates to use for maximizing over the constraint/objective
        weights (the analogue of the Lagrange multipliers). This parameter has
        *no effect* on the optimization of the model parameters. Defaults to
        "multiplicative".
      minimum_multiplier_radius: optional float in the range (0,1), only allowed
        if update_type="multiplicative". The constraint/objective weights will
        be lower bounded by "minimum_multiplier_radius" divided by one plus the
        number of constraints. Defaults to 10^-3.
      initial_multiplier_radius: optional float in the range [0,1], the initial
        value of each constraint weight (i.e. excluding the weight associated
        with the objective) will be "initial_multiplier_radius" divided by one
        plus the number of constraints. If update_type="multiplicative", it
        defaults to the value of minimum_multiplier_radius (and must be no
        smaller than minimum_multiplier_radius if it's explicitly specified),
        and defaults to zero if update_type="additive".

    Raises:
      ValueError: if the "regret_type" or "update_type" parameters are invalid,
        or if "minimum_multiplier_radius" or "initial_multiplier_radius" violate
        the conditions described above.
    """
    super(ProxyLagrangianOptimizer, self).__init__(optimizer=optimizer)
    self._constraint_optimizer = constraint_optimizer

    self._regret_type = regret_type.lower()
    if self._regret_type not in (_EXTERNAL_REGRET_TYPE, _SWAP_REGRET_TYPE):
      raise ValueError("regret_type must be either \"external\" or \"swap\"")

    self._update_type = update_type.lower()
    if self._update_type not in (_ADDITIVE_UPDATE_TYPE,
                                 _MULTPILICATIVE_UPDATE_TYPE):
      raise ValueError("update_type must be either \"additive\" or "
                       "\"multiplicative\"")

    if self._update_type == _ADDITIVE_UPDATE_TYPE:
      if minimum_multiplier_radius is not None:
        raise ValueError("minimum_multiplier_radius cannot be provided if "
                         "update_type is \"additive\"")
      if initial_multiplier_radius is None:
        initial_multiplier_radius = 0.0
      elif (initial_multiplier_radius < 0.0) or (initial_multiplier_radius >
                                                 1.0):
        raise ValueError("initial_multiplier_radius must be in the range [0,1]")
    else:
      if minimum_multiplier_radius is None:
        minimum_multiplier_radius = 1e-3
      elif (minimum_multiplier_radius <= 0.0) or (minimum_multiplier_radius >=
                                                  1.0):
        raise ValueError("minimum_multiplier_radius must be in the range (0,1)")
      if initial_multiplier_radius is None:
        initial_multiplier_radius = minimum_multiplier_radius
      elif (initial_multiplier_radius <
            minimum_multiplier_radius) or (initial_multiplier_radius > 1.0):
        raise ValueError("initial_multiplier_radius must be in the range "
                         "[minimum_multiplier_radius,1]")

    self._minimum_multiplier_radius = minimum_multiplier_radius
    self._initial_multiplier_radius = initial_multiplier_radius

    # We can't create the internal state here, since we don't know how many
    # constraints there will be until we see the ConstrainedMinimizationProblem.
    # Instead, we do so lazily, in _maybe_create_state().
    self._state = None

  @property
  def constraint_optimizer(self):
    """Returns the `Optimizer` used for the multipliers."""
    return self._constraint_optimizer

  def _maybe_create_state(self, num_constraints):
    """Fetches/creates the internal state."""
    # For swap regret minimizing updates, we have a total of m+1 weights, where
    # m is the number of constraints: one for the objective, and one for each
    # constraint.
    dimension = num_constraints + 1

    if self._state is not None:
      dims = self._state.shape.dims

      # We created the internal state below, and it should be impossible for it
      # to have an unknown shape or the wrong rank.
      assert dims is not None
      if self._regret_type == _EXTERNAL_REGRET_TYPE:
        assert len(dims) == 1
      else:
        assert len(dims) == 2

      # You can use this optimizer on multiple different problems (sharing the
      # internal state between them), but only if they have the same number of
      # constraints, and therefore the same internal state shape.
      if any(dim.value != dimension for dim in dims):
        raise RuntimeError(
            "if you use the same ProxyLagrangianOptimizer on multiple "
            "problems, then they must have the same number of constraints, so "
            "that the internal state (the analogues of the Lagrange "
            "multipliers) can be shared")

    else:
      # Initialize by putting as much weight as possible on the objective, and
      # as little as possible on the constraints.
      initial_one = 1.0 - (
          self._initial_multiplier_radius * (dimension - 1) / float(dimension))
      initial_zero = self._initial_multiplier_radius / float(dimension)
      # If using multiplicative updates, we won't store the distribution or
      # stochastic matrix itself: instead, we'll store its element-wise
      # logarithm.
      if self._update_type == _MULTPILICATIVE_UPDATE_TYPE:
        initial_one = math.log(initial_one)
        initial_zero = math.log(initial_zero)

      if self._regret_type == _EXTERNAL_REGRET_TYPE:
        # For external regret minimization, the internal state is shape-(m+1),
        # representing a multinoulli distribution (if using additive updates) or
        # its element-wise logarithm (if using multiplicative updates).
        #
        # FUTURE WORK: make the dtype a parameter.
        initial_state = np.concatenate(
            (np.full(shape=(1,), fill_value=initial_one, dtype=np.float32),
             np.full(
                 shape=(dimension - 1,),
                 fill_value=initial_zero,
                 dtype=np.float32)),
            axis=0)
      else:
        # This assertion should always succeed, since we check regret_type in
        # the constructor.
        assert self._regret_type == _SWAP_REGRET_TYPE

        # For swap regret minimization, the internal state is shape-(m+1,m+1),
        # representing a left-stochastic matrix (if using additive updates) or
        # its element-wise logarithm (if using multiplicative updates).
        #
        # FUTURE WORK: make the dtype a parameter.
        initial_state = np.concatenate((np.full(
            shape=(1, dimension), fill_value=initial_one, dtype=np.float32),
                                        np.full(
                                            shape=(dimension - 1, dimension),
                                            fill_value=initial_zero,
                                            dtype=np.float32)),
                                       axis=0)

      # FUTURE WORK: make the dtype a parameter.
      self._state = tf.Variable(
          initial_state,
          trainable=False,
          name="proxy_lagrangian_state",
          dtype=tf.float32,
          use_resource=True)

    return self._state

  def _distribution(self, state):
    """Returns the distribution represented by the internal state."""
    if self._regret_type == _EXTERNAL_REGRET_TYPE:
      if self._update_type == _ADDITIVE_UPDATE_TYPE:
        # If using additive external regret, the state is the distribution
        # itself.
        distribution = state
      else:
        # This assertion should always succeed, since we check update_type in
        # the constructor.
        assert self._update_type == _MULTPILICATIVE_UPDATE_TYPE

        # If using multiplicative external regret, the state is the element-wise
        # logarithm of the distribution.
        distribution = tf.exp(state)
    else:
      # This assertion should always succeed, since we check regret_type in the
      # constructor.
      assert self._regret_type == _SWAP_REGRET_TYPE

      if self._update_type == _ADDITIVE_UPDATE_TYPE:
        # If using additive swap regret, the state is a left-stochastic matrix
        # of which a maximum eigenvector is the distribution.
        stochastic_matrix = state
      else:
        # This assertion should always succeed, since we check update_type in
        # the constructor.
        assert self._update_type == _MULTPILICATIVE_UPDATE_TYPE

        # If using additive swap regret, the state is the element-wise logarithm
        # of the left-stochastic matrix of which a maximum eigenvector is the
        # distribution.
        stochastic_matrix = tf.exp(state)

      # The internal state used for swap regret minimization is a representation
      # of a left-stochastic shape-(m+1,m+1) `Tensor`, where m is the number of
      # constraints. A (m+1)-dimensional maximum eigenvector of this matrix
      # (i.e. a stationary distribution, since the matrix is left-stochastic)
      # represents the distribution over the objective function and the m
      # constraints w.r.t. which we're maximizing the proxy-Lagrangian
      # formulation.
      distribution = _maximal_eigenvector_power_method(stochastic_matrix)
      distribution = tf.abs(distribution)
      distribution /= tf.reduce_sum(distribution)

    return distribution

  def _state_gradient(self, zero_and_constraints, distribution):
    """Returns the gradient to apply to the internal state."""
    if self._regret_type == _EXTERNAL_REGRET_TYPE:
      return tf.cast(zero_and_constraints, distribution.dtype)
    else:
      # This assertion should always succeed, since we check regret_type in the
      # constructor.
      assert self._regret_type == _SWAP_REGRET_TYPE

      return tf.matmul(
          tf.expand_dims(tf.cast(zero_and_constraints, distribution.dtype), 1),
          tf.expand_dims(distribution, 0))

  def _projection_op(self, state, name):
    """Projects the internal state onto the feasible region."""
    with tf.colocate_with(state):
      if self._update_type == _ADDITIVE_UPDATE_TYPE:
        # If we're performing additive updates, then the feasible region is
        # either the space of multinoulli distributions (if minimizing external
        # regret) or the space of left-stochastic matrices (if minimizing swap
        # regret).
        return state.assign(
            _project_distribution_wrt_euclidean_norm(state), name=name)
      else:
        # This assertion should always succeed, since we check update_type in
        # the constructor.
        assert self._update_type == _MULTPILICATIVE_UPDATE_TYPE

        # Gets the dimension of the state (num_constraints + 1)--the assertions
        # are of things that cannot possibly fail.
        state_shape = state.shape.dims
        assert state_shape is not None
        dimension = state_shape[0].value
        assert dimension is not None

        minimum_log_multiplier = math.log(self._minimum_multiplier_radius /
                                          float(dimension))

        # If we're performing multiplicative updates, then the feasible region
        # is either the space element-wise logarithms of multinoulli
        # distributions (if minimizing external regret) or the space of
        # element-wise logarithms of left-stochastic matrices (if minimizing
        # swap regret). In either case, the log of the minimal element is also
        # constrained to be at least log(minimum_multiplier_radius / (m + 1)),
        # where m is the number of constraints.
        return state.assign(
            tf.maximum(
                _project_log_distribution_wrt_kl_divergence(state),
                tf.constant(minimum_log_multiplier, dtype=state.dtype)),
            name=name)

  def _minimize_constrained(self,
                            minimization_problem,
                            global_step=None,
                            var_list=None,
                            name=None):
    """Returns an `Operation` for minimizing the constrained problem.

    The "optimizer" constructor parameter will be used to update the model
    parameters, while the constraint/objective weights (the analogues of
    Lagrange multipliers) will be updated using "constraint_optimizer" (if
    provided) or "optimizer" (if not).

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
    # Create the internal state over which we will minimize swap regret.
    num_constraints = minimization_problem.num_constraints
    state = self._maybe_create_state(num_constraints)
    distribution = self._distribution(state)

    def loss_fn():
      """Evaluates the loss for the current internal state."""
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

      objective_and_proxy_constraints = tf.concat(
          (tf.expand_dims(objective, 0), proxy_constraints), axis=0)
      return tf.tensordot(
          tf.cast(distribution, objective_and_proxy_constraints.dtype),
          objective_and_proxy_constraints, 1)

    # Create a list of (gradient,variable) pairs for the model parameters. In
    # graph mode, Optimizer.compute_gradients() expects a Tensor, while in eager
    # mode, it expects a function returning a Tensor.
    if not tf.executing_eagerly():
      loss_fn = loss_fn()
    grads_and_vars = self.optimizer.compute_gradients(loss_fn, var_list)

    # Compute the gradient w.r.t. the internal state.
    constraints = minimization_problem.constraints()
    constraints = tf.reshape(constraints, shape=(num_constraints,))
    zero_and_constraints = tf.concat((tf.zeros(
        (1,), dtype=constraints.dtype), constraints),
                                     axis=0)
    state_gradient = self._state_gradient(zero_and_constraints, distribution)

    # Create a one-element list of (gradient,variable) pairs for the internal
    # state. We negate state_gradient since we want to maximize We negate
    # state_gradient since we want to maximize over the objective/constraint
    # weights.
    state_grads_and_vars = [(-state_gradient, state)]

    update_ops = []
    if self._constraint_optimizer is None:
      # If we don't have a separate constraint_optimizer, then we use
      # self._optimizer for both the update of the model parameters, and that of
      # the internal state.
      update_ops.append(
          self.optimizer.apply_gradients(
              grads_and_vars + state_grads_and_vars, name="update"))
    else:
      # If we have a separate constraint_optimizer, then we use self._optimizer
      # for the update of the model parameters, and self._constraint_optimizer
      # for that of the internal state.
      gradients = [
          gradient for gradient, _ in grads_and_vars + state_grads_and_vars
          if gradient is not None
      ]
      # We need this tf.control_dependencies() block so that the internal state
      # doesn't get updated before the model parameter gradients are computed.
      with tf.control_dependencies(gradients):
        update_ops.append(
            self.optimizer.apply_gradients(grads_and_vars, name="update"))
        update_ops.append(
            self._constraint_optimizer.apply_gradients(
                state_grads_and_vars, name="proxy_lagrangian_state_update"))

    with tf.control_dependencies(update_ops):
      if global_step is None:
        # If we don't have a global step, just project, and we're done.
        return self._projection_op(state, name=name)
      else:
        # If we have a global step, then we need to increment it in addition to
        # projecting.
        projection_op = self._projection_op(
            state, name="proxy_lagrangian_state_projection")
        with tf.colocate_with(global_step):
          global_step_op = global_step.assign_add(
              1, name="global_step_increment")
        return tf.group(projection_op, global_step_op, name=name)
