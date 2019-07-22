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
"""Contains the class representing a rate-constraints problem.

A rate constraints problem is constructed as an objective function and set of
constraints, all of which are represented as `Expression`s, each of which
represents a linear combination of `Term`s (which represent rates) and
`Tensor`s.

Rates are things like the error rate, the false positive rate, and so on. The
`RateMinimizationProblem` class defined represents the overall rate-based
optimization problem, and can be optimized using a `ConstrainedOptimizer`.

Example
=======

Suppose that "examples_tensor" and "labels_tensor" are placeholder `Tensor`s
containing training examples and their associated labels, and the "model"
function evaluates the model you're trying to train, returning a `Tensor` of
model evaluations. Further suppose that we wish to impose a fairness-style
constraint on a protected class (the "blue" examples). The following code will
create two contexts (see subsettable_context.py) representing all of the
examples, and only the "blue" examples, respectively.

>>> ctx = rate_context(model(examples_tensor), labels_tensor)
>>> blue_ctx = ctx.subset(examples_tensor[:, is_blue_idx])

Now that we have the contexts, we can create the rates. We'll try to minimize
the overall error rate, while constraining the true positive rate on the "blue"
class to be between 90% and 110% of the overall true positive rate:

>>> objective = error_rate(ctx)
>>> constraints = [
>>>     true_positive_rate(blue_ctx) >= 0.9 * true_positive_rate(ctx),
>>>     true_positive_rate(blue_ctx) <= 1.1 * true_positive_rate(ctx)
>>> ]

The error_rate() and true_positive_rate() functions are two of the
rate-constructing functions that are defined rates.py. This objective and
list of constraints can then be passed on to the `RateMinimizationProblem`
constructor as follows:

>>> problem = RateMinimizationProblem(objective, constraints)

This problem can then be solved using a `ConstrainedOptimizer`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_constrained_optimization.python import constrained_minimization_problem

from tensorflow_constrained_optimization.python.rates import basic_expression


class RateMinimizationProblem(
    constrained_minimization_problem.ConstrainedMinimizationProblem):
  """A rate constrained optimization problem.

  In order to optimize a rate constrained problem, you should construct an
  instance of this class from the objective function that you wish to minimize
  and the list of constraints you wish to impose. This class implements the
  `ConstrainedMinimizationProblem` interface, so the resulting object can then
  be optimized using a `ConstrainedOptimizer`.

  It's important to understand that this object is *stateful*. The denominators
  of the rates are estimated from running sums over all of the data that have
  been seen by pre_train_ops (which will be executed at the start of every
  train_op). Hence, these estimated denominators--upon which the objective,
  constraints and proxy_constraints `Tensor`s depend--will change from
  iteration-to-iteration as they converge to their true values. This state can
  be re-initialized by executing the restart_ops.
  """

  def __init__(self, objective, constraints=None, denominator_lower_bound=1e-3):
    """Creates a rate constrained optimization problem.

    In addition to an objective function to minimize and a list of constraints
    to satisfy, this method also takes a "denominator_lower_bound" parameter. At
    a high level, a rate is "the proportion of training examples satisfying some
    property for which some event occurs, divided by the proportion of training
    examples satisfying the property", i.e. is a numerator divided by a
    denominator. To avoid dividing by zero (or quantities that are close to
    zero), the "denomintor_lower_bound" parameter is used to impose a lower
    bound on the denominator of a rate. However, this parameter is a last
    resort. If you're calculating a rate on a small subset of the data (i.e.
    with a property that is rately true, resulting in a small denominator), then
    the speed of optimization could suffer greatly: you'd almost certainly be
    better off splitting the subset of interest off into its own dataset, with
    its own placeholder tensors.

    Args:
      objective: an `Expression` to minimize.
      constraints: a collection of `Constraint`s to impose.
      denominator_lower_bound: float, the smallest permitted value of the
        denominator of a rate.

    Raises:
      RuntimeError: if we're running in eager mode.
      ValueError: if the "penalty" portion of the objective or a constraint is
        non-differentiable, or if denominator_lower_bound is negative.
    """
    if tf.executing_eagerly():
      raise RuntimeError("the TFCO rate helpers cannot be used in eager mode")

    # We do permit denominator_lower_bound to be zero. In this case, division by
    # zero is possible, and it's the user's responsibility to ensure that it
    # doesn't happen.
    if denominator_lower_bound < 0.0:
      raise ValueError("denominator lower bound must be nonnegative")
    # The objective needs to be differentiable. So do the penalty portions of
    # the constraints, but we'll check those later.
    if not objective.penalty_expression.is_differentiable:
      raise ValueError("non-differentiable losses (e.g. the zero-one loss) "
                       "cannot be optimized--they can only be constrained")

    if constraints is None:
      constraints = set()
    else:
      constraints = set(constraints)

    # We make our own global_step, for keeping track of the denominators. We
    # don't take one as a parameter since we want complete ownership, to avoid
    # any shenanigans: it has to start at zero, and be incremented after every
    # minibatch.
    global_step = tf.Variable(
        0, trainable=False, dtype=tf.int64, name="global_step")

    # This evaluation context will remember and re-use certain intermediate
    # values, causing the TensorFlow graph we construct to contain fewer
    # redundancies than it would otherwise.
    evaluation_context = basic_expression.BasicExpression.EvaluationContext(
        denominator_lower_bound, global_step)

    # We ignore the "constraint_expression" field here, since we're not inside a
    # constraint (this is the objective function).
    self._objective, pre_train_ops, restart_ops = (
        objective.penalty_expression.evaluate(evaluation_context))
    constraints.update(objective.extra_constraints)

    # Evaluating expressions can result in extra constraints being introduced,
    # so we keep track of constraints that we've already evaluated in
    # "checked_constraints", add new constraints to "constraints", and
    # repeatedly evaluate the contents of constraints - checked_constraints
    # until none are left.
    checked_constraints = set()
    penalty_values = []
    constraint_values = []
    while constraints != checked_constraints:
      new_constraints = constraints - checked_constraints
      for constraint in new_constraints:
        if not constraint.expression.penalty_expression.is_differentiable:
          raise ValueError("non-differentiable losses (e.g. the zero-one loss) "
                           "cannot be optimized--they can only be constrained")
        penalty_value, penalty_pre_train_ops, penalty_restart_ops = (
            constraint.expression.penalty_expression.evaluate(
                evaluation_context))
        constraint_value, constraint_pre_train_ops, constraint_restart_ops = (
            constraint.expression.constraint_expression.evaluate(
                evaluation_context))
        penalty_values.append(penalty_value)
        constraint_values.append(constraint_value)
        pre_train_ops.update(penalty_pre_train_ops)
        pre_train_ops.update(constraint_pre_train_ops)
        restart_ops.update(penalty_restart_ops)
        restart_ops.update(constraint_restart_ops)
        constraints.update(constraint.expression.extra_constraints)
      checked_constraints.update(new_constraints)

    self._num_constraints = len(penalty_values)
    assert self._num_constraints == len(constraint_values)

    self._proxy_constraints = tf.stack(penalty_values)
    self._constraints = tf.stack(constraint_values)

    # Increment our internal global_step after all of the other pre_train_ops.
    with tf.control_dependencies(pre_train_ops):
      self._pre_train_ops = [tf.assign_add(global_step, 1)]
    # Add an op that re-initializes the global_step to restart_ops.
    restart_ops.add(tf.assign(global_step, 0))
    self._restart_ops = list(restart_ops)

  def objective(self):
    """Returns the objective function.

    Returns:
      A scalar `Tensor` that should be minimized.
    """
    return self._objective

  @property
  def num_constraints(self):
    """Returns the number of constraints.

    Returns:
      An int containing the number of constraints.
    """
    return self._num_constraints

  def constraints(self):
    """Returns the `Tensor` of constraint functions.

    The constraints are element-wise non-positivity constraints, i.e. if g_i is
    the ith element of the constraints `Tensor`, then the ith constraint is g_i
    <= 0.

    Returns:
      A rank-1 `Tensor` of constraint functions.
    """
    return self._constraints

  def proxy_constraints(self):
    """Returns the optional `Tensor` of proxy constraint functions.

    The difference between `constraints` and `proxy_constraints` is that the
    former are used when updating the Lagrange multipliers (or equivalent),
    while the latter are used for the model update.

    Returns:
      A rank-1 `Tensor` of proxy constraint functions.
    """
    return self._proxy_constraints

  def pre_train_ops(self):
    """Returns a list of `tf.Operation`s to run at the start of train_op.

    When a constrained optimizer creates a train_op, it will include these ops
    before the main training step. These ops update internal state that is used
    to estimate the denominators of rates.

    Returns:
      A list of `tf.Operation`s.
    """
    return self._pre_train_ops[:]

  def restart_ops(self):
    """Returns a list of `tf.Operation`s that restart the pre_train_ops.

    Executing the pre_train_ops updates the internal state used to keep track of
    the denominators of the rates. The restart_ops returned by this method can
    be executed to re-initialize this state, if you want to "start over".

    Returns:
      A list of `tf.Operation`s.
    """
    return self._restart_ops[:]
