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
"""Constrained optimization using the proxy-Lagrangian formulation.

The code in this file uses the proxy-Lagrangian formulation to minimize a
constrained problem specified using the `ConstrainedMinimizationProblem`
interface. Compared with the Lagrangian formulation, the proxy-Lagrangian is the
more complex and expensive option, although it has more theoretical backing.
Generally, the extra computational cost of using the proxy-Lagrangian
formuilation will be small, assuming that the minibatch size and complexity of
the model are sufficiently large compared to the number of constraints. In this
case, we recommend it over the Lagrangian formulation.

At least in theory, external regret minimization of the Lagrangian formulation
could converge more slowly than swap regret minimization using the
proxy-Lagrangian formulation, assuming that the `ConstrainedMinimizationProblem`
we're optimizing uses "proxy_constraints".

For more specifics, please refer to:

> Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
> Constrained Optimization". ALT'19.
> [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

The proxy-Lagrangian formulation can be found in Definition 2, and is discussed
in Section 4. If using multiplicative updates to minimize swap regret (the
update_type and regret_type constructor parameters), the algorithm is most
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
import tensorflow.compat.v2 as tf

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
    epsilon: non-negative float, if two iterations of the power method differ
      (in L2 norm) by no more than epsilon, we will terminate.
    maximum_iterations: non-negative int, if we perform this many iterations, we
      will terminate.

  Returns:
    A maximal right-eigenvector of "matrix".

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
  distribution_dims = distribution.shape.dims
  if distribution_dims is None:
    raise ValueError("distribution must have a known rank")
  dimension = distribution_dims[0].value
  if dimension is None:
    raise ValueError("distribution must have a known first dimension")

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
  log_distribution = log_distribution - tf.math.log(
      tf.reduce_sum(tf.exp(log_distribution), axis=0, keepdims=True))
  return log_distribution


class _ProxyLagrangianFormulation(constrained_optimizer.Formulation):
  """Contains the internal state for the proxy-Lagrangian formulation.

  To optimize using the proxy-Lagrangian formulation, we jointly minimize over
  the model parameters, and maximize over constraint/objective weights (the
  analogues of Lagrange multipliers). This latter maximization uses either
  multiplicative or additive updates, and either an algorithm that minimizes
  swap regret, or one that minimizes external regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer is the so-called "proxy-Lagrangian"
  formulation, which can be found in Definition 2.
  """

  def __init__(self,
               regret_type,
               update_type,
               minimum_multiplier_radius=None,
               initial_multiplier_radius=None,
               dual_scale=1.0,
               variable_fn=tf.Variable):
    """Constructs a new `_ProxyLagrangianFormulation`.

    Args:
      regret_type: string, either "external" or "swap", determines what type of
        regret minimization to perform when updating the constraint/objective
        weights (the analogues of the Lagrange multipliers). This parameter has
        *no effect* on the optimization of the model parameters. Defaults to
        "swap".
      update_type: string, either "additive" or "multiplicative", determines
        what type of updates to use for maximizing over the constraint/objective
        weights (the analogues of the Lagrange multipliers). This parameter has
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
      dual_scale: optional float defaulting to 1, a multiplicative scaling
        factor applied to gradients w.r.t. the internal state.
      variable_fn: optional function with the same signature as the
        `tf.Variable` constructor, that returns a new variable with the
        specified properties.

    Raises:
      ValueError: if the "regret_type" or "update_type" parameters are invalid,
        if "minimum_multiplier_radius" or "initial_multiplier_radius" violate
        the conditions described above, or if "dual_scale" is non-positive.
    """
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

    if dual_scale <= 0.0:
      raise ValueError("dual_scale must be strictly positive")
    self._dual_scale = dual_scale

    self._variable_fn = variable_fn

    # We can't create the internal state here, since we don't know how many
    # constraints there will be until we see the ConstrainedMinimizationProblem.
    # Instead, we do so lazily, in create_state().
    self._state = None

  @property
  def state(self):
    return self._state

  def _project_state(self, state):
    """Projects the internal state onto the feasible region."""
    if self._update_type == _ADDITIVE_UPDATE_TYPE:
      # If we're performing additive updates, then the feasible region is either
      # the space of multinoulli distributions (if minimizing external regret)
      # or the space of left-stochastic matrices (if minimizing swap regret).
      return _project_distribution_wrt_euclidean_norm(state)
    else:
      # This assertion should always succeed, since we check update_type in the
      # constructor.
      assert self._update_type == _MULTPILICATIVE_UPDATE_TYPE

      # Gets the dimension of the state (num_constraints + 1)--the assertions
      # are of things that cannot possibly fail.
      state_dims = state.shape.dims
      assert state_dims is not None
      dimension = state_dims[0].value
      assert dimension is not None

      minimum_log_multiplier = math.log(self._minimum_multiplier_radius /
                                        float(dimension))

      # If we're performing multiplicative updates, then the feasible region is
      # either the space element-wise logarithms of multinoulli distributions
      # (if minimizing external regret) or the space of element-wise logarithms
      # of left-stochastic matrices (if minimizing swap regret). In either case,
      # the log of the minimal element is also constrained to be at least
      # log(minimum_multiplier_radius / (m + 1)), where m is the number of
      # constraints.
      return tf.maximum(
          _project_log_distribution_wrt_kl_divergence(state),
          tf.constant(minimum_log_multiplier, dtype=state.dtype))

  def create_state(self, num_constraints):
    """Fetches (or creates, if necessary) the internal state."""
    # For swap regret minimizing updates, we have a total of m+1 weights, where
    # m is the number of constraints: one for the objective, and one for each
    # constraint.
    dimension = num_constraints + 1

    if self._state is not None:
      # We created the internal state below, and it should be impossible for it
      # to have an unknown shape or the wrong rank.
      state_dims = self._state.shape.dims
      assert state_dims is not None
      if self._regret_type == _EXTERNAL_REGRET_TYPE:
        assert len(state_dims) == 1
      else:
        assert len(state_dims) == 2

      # You can use this optimizer on multiple different problems (sharing the
      # internal state between them), but only if they have the same number of
      # constraints, and therefore the same internal state shape.
      if any(dim.value != dimension for dim in state_dims):
        raise RuntimeError(
            "if you use the same proxy-Lagrangian optimizer on multiple "
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
      self._state = self._variable_fn(
          initial_value=initial_state,
          trainable=True,
          name="tfco_proxy_lagrangian_state",
          dtype=tf.float32,
          constraint=self._project_state)

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

      return tf.tensordot(
          tf.cast(zero_and_constraints, distribution.dtype),
          distribution,
          axes=0)

  def get_loss_fn(self, minimization_problem):
    """Returns the proxy-Lagrangian loss function.

    The resulting loss function uses `tf.custom_gradient` to override its
    gradients. In particular, the gradients w.r.t. the internal state will be
    negated (since we wish to maximize them), and will be written in terms of
    the constraints, instead of the proxy_constraints.

    Args:
      minimization_problem: `ConstrainedMinimizationProblem`, the problem to
        minimize.

    Returns:
      The proxy-Lagrangian loss function.
    """
    # This function returns both the value of the Lagrangian, and its gradients
    # w.r.t. the contents of the constrained minimization problem (objective,
    # constraints and proxy_constraints) and the internal proxy-Lagrangian
    # state (state). The reason for using tf.custom_gradient is that it allows
    # us to override the gradient w.r.t. the state to use (i) the original
    # constraints instead of the proxy constraints, and (ii) to create swap
    # regret updates, if necessary.
    @tf.custom_gradient
    def loss_gradient_fn(objective, constraints, proxy_constraints, state):
      """Evaluates the loss for the current internal state."""
      # Make sure that the objective and proxy constraints have the same dtype.
      if (constraints.dtype.base_dtype != objective.dtype.base_dtype or
          proxy_constraints.dtype.base_dtype != objective.dtype.base_dtype):
        raise TypeError("objective, constraints and proxy_constraints must all "
                        "have the same dtypes")

      distribution = self._distribution(state)
      zero_and_constraints = tf.pad(constraints, [[1, 0]])

      # The proxy-Lagrangian is defined as:
      #   objective_and_proxy_constraints = tf.concat(
      #       (tf.expand_dims(objective, 0), proxy_constraints), axis=0)
      #   output = tf.tensordot(
      #       distribution, objective_and_proxy_constraints, axes=1)
      # when updating the model parameters, and as:
      #   output = tf.tensordot(
      #       distribution, zero_and_constraints, axes=1)
      # when updating the internal state.
      #
      # However, while these quantities are what we differentiate, we do *not*
      # return either of their values (since they aren't terribly meaningful,
      # thanks to being simultaneously minimized over the model parameters and
      # maximized over the internal state). Instead, we just return the value
      # of the objective.
      output = objective

      wrt_objective = tf.cast(distribution[0], dtype=objective.dtype.base_dtype)
      wrt_proxy_constraints = tf.cast(
          distribution[1:], dtype=proxy_constraints.dtype.base_dtype)
      wrt_state = -self._state_gradient(zero_and_constraints, distribution)

      def gradient_fn(output_gradient):
        # We return the gradient w.r.t. the objective, constraints,
        # proxy_constraints and internal state, respectively (this is the same
        # order as the arguments to loss_gradient_fn). Notice that the gradient
        # w.r.t. the constraints is None, and that w.r.t. the internal state is
        # scaled by dual_scale.
        return (output_gradient * wrt_objective, None,
                output_gradient * wrt_proxy_constraints,
                self._dual_scale * output_gradient * wrt_state)

      return output, gradient_fn

    # Create the internal state.
    num_constraints = minimization_problem.num_constraints
    state = self.create_state(num_constraints)

    # We don't use functools.partial since we need the arguments to be evaluated
    # when the loss is called, not when we construct the partial application.
    def partial_loss_gradient_fn():
      objective, constraints, proxy_constraints = (
          minimization_problem.components())
      constraints = tf.reshape(constraints, shape=(num_constraints,))
      if proxy_constraints is None:
        proxy_constraints = constraints
      else:
        proxy_constraints = tf.reshape(
            proxy_constraints, shape=(num_constraints,))
      return loss_gradient_fn(objective, constraints, proxy_constraints, state)

    return partial_loss_gradient_fn


def create_proxy_lagrangian_loss(minimization_problem,
                                 regret_type=_SWAP_REGRET_TYPE,
                                 update_type=_MULTPILICATIVE_UPDATE_TYPE,
                                 minimum_multiplier_radius=None,
                                 initial_multiplier_radius=None,
                                 dual_scale=1.0,
                                 variable_fn=tf.Variable):
  """Creates a loss function from a `ConstrainedMinimizationProblem`.

  Minimizing the returned loss will have the effect of jointly minimizing
  over the model parameters, and maximizing over constraint/objective weights
  (the analogues of Lagrange multipliers). This latter maximization uses either
  multiplicative or additive updates, and either an algorithm that minimizes
  swap regret, or one that minimizes external regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used here is the so-called "proxy-Lagrangian"
  formulation, which can be found in Definition 2, and is discussed in Section
  4. If using multiplicative updates to minimize swap regret (the update_type
  and regret_type constructor parameters), the algorithm is most similar to
  Algorithm 2 in Section 4, with the difference being that it uses
  `Optimizer`s, instead of SGD, for the "inner" updates. Instead of swap regret,
  one can alternatively use external regret, and instead of multiplicative
  updates, one can alternatively use additive updates, if desired.

  In addition to a loss function, this method returns a function returning a
  list of operations that should be executed before each iteration
  ("update_ops"), and a variable containing the internal proxy-Lagrangian state
  (the analogue of the Lagrange multipliers).

  In graph mode, the result of this update_ops function could be "attached" to
  the train_op using tf.control_dependencies. In eager mode, it should be called
  before each iteration. Likewise, you should make sure to differentiate w.r.t.
  the state, e.g. by including it in the var_list that you pass to an
  `Optimizer`'s minimize() method.

  For example, in graph mode, your code could look like this:

  ```python
  # We ignore the returned state, since we won't provide a var_list to the
  # Optimizer's minimize() method.
  loss_fn, update_ops_fn, _ = create_loss(formulation, minimization_problem)

  optimizer = tf.compat.v1.train.GradientDescentOptimizer()
  with tf.control_dependencies(update_ops_fn()):
    # Since we don't provide a var_list, we'll just differentiate w.r.t. all
    # trainable variables, including the state.
    train_op = optimizer.minimize(loss_fn())

  with tf.compat.v1.Session() as session:
    for iteration in xrange(num_iterations):
      session.run(train_op)
  ```

  while in eager mode, it could look like this:

  ```python
  loss_fn, update_ops_fn, state_variable = create_loss(formulation,
      minimization_problem)

  # Assuming that we already have a var_list containing the model parameters.
  var_list += minimization_problem.trainable_variables
  var_list.append(state_variable)

  optimizer = tf.keras.optimizers.SGD()

  for iteration in xrange(num_iterations):
    update_ops_fn()
    optimizer.minimize(loss_fn, var_list=var_list)
  ```

  Args:
    minimization_problem: `ConstrainedMinimizationProblem`, the problem to
      optimize.
    regret_type: string, either "external" or "swap", determines what type of
      regret minimization to perform when updating the constraint/objective
      weights (the analogues of the Lagrange multipliers). This parameter has
      *no effect* on the optimization of the model parameters. Defaults to
      "swap".
    update_type: string, either "additive" or "multiplicative", determines what
      type of updates to use for maximizing over the constraint/objective
      weights (the analogues of the Lagrange multipliers). This parameter has
      *no effect* on the optimization of the model parameters. Defaults to
      "multiplicative".
    minimum_multiplier_radius: optional float in the range (0,1), only allowed
      if update_type="multiplicative". The constraint/objective weights will be
      lower bounded by "minimum_multiplier_radius" divided by one plus the
      number of constraints. Defaults to 10^-3.
    initial_multiplier_radius: optional float in the range [0,1], the initial
      value of each constraint weight (i.e. excluding the weight associated with
      the objective) will be "initial_multiplier_radius" divided by one plus the
      number of constraints. If update_type="multiplicative", it defaults to the
      value of minimum_multiplier_radius (and must be no smaller than
      minimum_multiplier_radius if it's explicitly specified), and defaults to
      zero if update_type="additive".
    dual_scale: optional float defaulting to 1, a multiplicative scaling factor
      applied to gradients w.r.t. the internal state.
    variable_fn: optional function with the same signature as the `tf.Variable`
      constructor, that returns a new variable with the specified properties.

  Returns:
    A (loss_fn, update_ops_fn, state_variable) tuple, where loss_fn is a nullary
    function returning a `Tensor` that can be minimized to optimize the
    constrained problem, update_ops_fn is a nullary function that returns a list
    of operations that should be executed before each training iteration, and
    state_variable is a variable containing the internal state of the
    proxy-Lagrangian formulation.
  """
  return constrained_optimizer.create_loss(
      _ProxyLagrangianFormulation(
          regret_type=regret_type,
          update_type=update_type,
          minimum_multiplier_radius=minimum_multiplier_radius,
          initial_multiplier_radius=initial_multiplier_radius,
          dual_scale=dual_scale,
          variable_fn=variable_fn), minimization_problem)


class ProxyLagrangianOptimizerV1(constrained_optimizer.ConstrainedOptimizerV1):
  """A `ConstrainedOptimizerV1` based on the proxy-Lagrangian formulation.

  This constrained optimizer uses the given `tf.compat.v1.train.Optimizer`s to
  jointly minimize over the model parameters, and maximize over
  constraint/objective weights (the analogues of Lagrange multipliers). This
  latter maximization uses either multiplicative or additive updates, and either
  an algorithm that minimizes swap regret, or one that minimizes external
  regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer is the so-called "proxy-Lagrangian"
  formulation, which can be found in Definition 2, and is discussed in Section
  4. If using multiplicative updates to minimize swap regret (the update_type
  and regret_type constructor parameters), the algorithm is most similar to
  Algorithm 2 in Section 4, with the difference being that it uses
  `tf.compat.v1.train.Optimizer`s, instead of SGD, for the "inner" updates.
  Instead of swap regret, one can alternatively use external regret, and instead
  of multiplicative updates, one can alternatively use additive updates, if
  desired.

  The internal state (the analogues of the Lagrange multipliers) are owned by
  this optimizer. Hence, if you want to use a `ProxyLagrangianOptimizerV1` on
  multiple `ConstrainedMinimizationProblem`s, while sharing this internal state
  between them, then you may do so. However, each problem must have the same
  number of constraints (an exception will be raised, otherwise), so that the
  internal states are compatible.
  """

  def __init__(self,
               optimizer,
               num_constraints=None,
               constraint_optimizer=None,
               regret_type=_SWAP_REGRET_TYPE,
               update_type=_MULTPILICATIVE_UPDATE_TYPE,
               minimum_multiplier_radius=None,
               initial_multiplier_radius=None,
               name="ProxyLagrangianOptimizerV1"):
    """Constructs a new `ProxyLagrangianOptimizerV1`.

    The difference between "optimizer" and "constraint_optimizer" (if the latter
    is provided) is that the former is used for learning the model parameters,
    while the latter us used for the update to the constraint/objective weights
    (the analogues of Lagrange multipliers). If no "constraint_optimizer" is
    provided, then "optimizer" is used for both.

    Args:
      optimizer: `tf.compat.v1.train.Optimizer`, used to optimize the objective
        and proxy_constraints portion of `ConstrainedMinimizationProblem`. If
        constraint_optimizer is not provided, this will also be used to optimize
        the internal constrained optimization state (the analogues of the
        Lagrange multipliers).
      num_constraints: optional int, the number of constraints in the
        `ConstrainedMinimizationProblem` that will eventually be minimized. If
        this argument is provided, then the internal state will be created
        inside this constructor. Otherwise, it will be created inside the first
        call to get_loss_fn().
      constraint_optimizer: optional `tf.compat.v1.train.Optimizer`, used to
        optimize the internal constrained optimization state (the analogues of
        the Lagrange multipliers).
      regret_type: string, either "external" or "swap", determines what type of
        regret minimization to perform when updating the constraint/objective
        weights (the analogues of the Lagrange multipliers). This parameter has
        *no effect* on the optimization of the model parameters. Defaults to
        "swap".
      update_type: string, either "additive" or "multiplicative", determines
        what type of updates to use for maximizing over the constraint/objective
        weights (the analogues of the Lagrange multipliers). This parameter has
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
      name: as in `ConstrainedOptimizerV1`.

    Raises:
      ValueError: if the "regret_type" or "update_type" parameters are invalid,
        or if "minimum_multiplier_radius" or "initial_multiplier_radius" violate
        the conditions described above.
    """
    super(ProxyLagrangianOptimizerV1, self).__init__(
        _ProxyLagrangianFormulation(
            regret_type=regret_type,
            update_type=update_type,
            minimum_multiplier_radius=minimum_multiplier_radius,
            initial_multiplier_radius=initial_multiplier_radius),
        optimizer=optimizer,
        num_constraints=num_constraints,
        constraint_optimizer=constraint_optimizer,
        name=name)


class ProxyLagrangianOptimizerV2(constrained_optimizer.ConstrainedOptimizerV2):
  """A `ConstrainedOptimizerV2` based on the proxy-Lagrangian formulation.

  This constrained optimizer uses the given `tf.keras.optimizers.Optimizer`s to
  jointly minimize over the model parameters, and maximize over
  constraint/objective weights (the analogues of Lagrange multipliers). This
  latter maximization uses either multiplicative or additive updates, and either
  an algorithm that minimizes swap regret, or one that minimizes external
  regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization". ALT'19.
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer is the so-called "proxy-Lagrangian"
  formulation, which can be found in Definition 2, and is discussed in Section
  4. If using multiplicative updates to minimize swap regret (the update_type
  and regret_type constructor parameters), the algorithm is most similar to
  Algorithm 2 in Section 4, with the difference being that it uses
  `tf.keras.optimizers.Optimizer`s, instead of SGD, for the "inner" updates.
  Instead of swap regret, one can alternatively use external regret, and instead
  of multiplicative updates, one can alternatively use additive updates, if
  desired.

  The internal state (the analogues of the Lagrange multipliers) are owned by
  this optimizer. Hence, if you want to use a `ProxyLagrangianOptimizerV2` on
  multiple `ConstrainedMinimizationProblem`s, while sharing this internal state
  between them, then you may do so. However, each problem must have the same
  number of constraints (an exception will be raised, otherwise), so that the
  internal states are compatible.
  """

  def __init__(self,
               optimizer,
               num_constraints,
               constraint_optimizer=None,
               regret_type=_SWAP_REGRET_TYPE,
               update_type=_MULTPILICATIVE_UPDATE_TYPE,
               minimum_multiplier_radius=None,
               initial_multiplier_radius=None,
               name="ProxyLagrangianOptimizer"):
    """Constructs a new `ProxyLagrangianOptimizerV2`.

    The difference between "optimizer" and "constraint_optimizer" (if the latter
    is provided) is that the former is used for learning the model parameters,
    while the latter us used for the update to the constraint/objective weights
    (the analogues of Lagrange multipliers). If no "constraint_optimizer" is
    provided, then "optimizer" is used for both.

    Args:
      optimizer: `tf.keras.optimizers.Optimizer`, used to optimize the objective
        and proxy_constraints portion of `ConstrainedMinimizationProblem`. If
        constraint_optimizer is not provided, this will also be used to optimize
        the internal constrained optimization state (the analogues of the
        Lagrange multipliers).
      num_constraints: int, the number of constraints in the
        `ConstrainedMinimizationProblem` that will eventually be minimized.
      constraint_optimizer: optional `tf.keras.optimizers.Optimizer`, used to
        optimize the internal constrained optimization state (the analogues of
        the Lagrange multipliers).
      regret_type: string, either "external" or "swap", determines what type of
        regret minimization to perform when updating the constraint/objective
        weights (the analogues of the Lagrange multipliers). This parameter has
        *no effect* on the optimization of the model parameters. Defaults to
        "swap".
      update_type: string, either "additive" or "multiplicative", determines
        what type of updates to use for maximizing over the constraint/objective
        weights (the analogues of the Lagrange multipliers). This parameter has
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
      name: as in `ConstrainedOptimizerV1`.

    Raises:
      ValueError: if the "regret_type" or "update_type" parameters are invalid,
        or if "minimum_multiplier_radius" or "initial_multiplier_radius" violate
        the conditions described above.
    """
    super(ProxyLagrangianOptimizerV2, self).__init__(
        _ProxyLagrangianFormulation(
            regret_type=regret_type,
            update_type=update_type,
            minimum_multiplier_radius=minimum_multiplier_radius,
            initial_multiplier_radius=initial_multiplier_radius),
        optimizer=optimizer,
        num_constraints=num_constraints,
        constraint_optimizer=constraint_optimizer,
        name=name)
