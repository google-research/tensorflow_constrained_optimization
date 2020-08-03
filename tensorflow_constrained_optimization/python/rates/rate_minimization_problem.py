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
optimization problem, and can be optimized using a `ConstrainedOptimizerV1` or
`ConstrainedOptimizerV2`.

Example
=======

Suppose that "examples_tensor" and "labels_tensor" are placeholder `Tensor`s
containing training examples and their associated labels, and the "model"
function evaluates the model you're trying to train, returning a `Tensor` of
model evaluations. Further suppose that we wish to impose a fairness-style
constraint on a protected class (the "blue" examples). The following code will
create two contexts (see subsettable_context.py) representing all of the
examples, and only the "blue" examples, respectively.

```python
ctx = rate_context(model(examples_tensor), labels_tensor)
blue_ctx = ctx.subset(examples_tensor[:, is_blue_idx])
```

Now that we have the contexts, we can create the rates. We'll try to minimize
the overall error rate, while constraining the true positive rate on the "blue"
class to be between 90% and 110% of the overall true positive rate:

```python
objective = error_rate(ctx)
constraints = [
    true_positive_rate(blue_ctx) >= 0.9 * true_positive_rate(ctx),
    true_positive_rate(blue_ctx) <= 1.1 * true_positive_rate(ctx)
]
```

The error_rate() and true_positive_rate() functions are two of the
rate-constructing functions that are defined rates.py. This objective and
list of constraints can then be passed on to the `RateMinimizationProblem`
constructor as follows:

```python
problem = RateMinimizationProblem(objective, constraints)
```

This problem can then be solved using a constrained optimizer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python import constrained_minimization_problem
from tensorflow_constrained_optimization.python.rates import constraint
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor


class RateMinimizationProblem(
    constrained_minimization_problem.ConstrainedMinimizationProblem):
  """A rate constrained optimization problem.

  In order to optimize a rate constrained problem, you should construct an
  instance of this class from the objective function that you wish to minimize
  and the list of constraints you wish to impose. This class implements the
  `ConstrainedMinimizationProblem` interface, so the resulting object can then
  be optimized using a `ConstrainedOptimizerV1` or `ConstrainedOptimizerV2`.

  It's important to understand that this object is *stateful*. In addition to
  slack variables (which are automatically inserted for certain rates), the
  denominators of the rates are estimated from running sums over all of the data
  that has been seen by update_ops (which will be executed at the start of every
  train_op). Hence, these estimated denominators--upon which the objective,
  constraints and proxy_constraints `Tensor`s depend--will change from
  iteration-to-iteration as they converge to their true values.
  """

  def __init__(self,
               objective,
               constraints=None,
               denominator_lower_bound=1e-3,
               variable_fn=tf.Variable):
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
      variable_fn: optional function with the same signature as the
        `tf.Variable` constructor, that returns a new variable with the
        specified properties.

    Raises:
      ValueError: if the "penalty" portion of the objective or a constraint is
        non-differentiable, or if denominator_lower_bound is negative.
    """
    # We do permit denominator_lower_bound to be zero. In this case, division by
    # zero is possible, and it's the user's responsibility to ensure that it
    # doesn't happen.
    if denominator_lower_bound < 0.0:
      raise ValueError("denominator lower bound must be non-negative")
    # The objective needs to be differentiable. So do the penalty portions of
    # the constraints, but we'll check those later.
    if not objective.penalty_expression.is_differentiable:
      raise ValueError("non-differentiable losses (e.g. the zero-one loss) "
                       "cannot be optimized--they can only be constrained")

    inputs = deferred_tensor.DeferredTensorInputList()
    variables = deferred_tensor.DeferredVariableList()
    constraints = constraint.ConstraintList(constraints)

    # We make our own global_step, for keeping track of the denominators. We
    # don't take one as a parameter since we want complete ownership, to avoid
    # any shenanigans: it has to start at zero, and be incremented after every
    # minibatch.
    self._global_step = variable_fn(
        0,
        trainable=False,
        name="tfco_global_step",
        dtype=tf.int64,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    # This structure_memoizer will remember and re-use certain intermediate
    # values, causing the TensorFlow graph we construct to contain fewer
    # redundancies than it would otherwise. Additionally, it will store any
    # slack variables or denominator variables that need to be created for the
    # optimization problem.
    self._structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: denominator_lower_bound,
        defaults.GLOBAL_STEP_KEY: self._global_step,
        defaults.VARIABLE_FN_KEY: variable_fn
    }

    # We ignore the "constraint_expression" field here, since we're not inside a
    # constraint (this is the objective function).
    self._objective = objective.penalty_expression.evaluate(
        self._structure_memoizer)
    inputs += self._objective.inputs
    variables += self._objective.variables
    constraints += objective.extra_constraints

    # Evaluating expressions can result in extra constraints being introduced,
    # so we keep track of the number of constraints that we've already evaluated
    # in "checked_constraints", append new constraints to "constraints" (which
    # will automatically ignore attempts to add duplicates, since it's a
    # ConstraintList), and repeatedly check the newly-added constraints until
    # none are left.
    #
    # In light of the fact that constraints can depend on other constraints, we
    # can view the structure of constraints as a tree, in which case this code
    # will enumerate over the constraints in breadth-first order.
    self._proxy_constraints = []
    self._constraints = []
    checked_constraints = 0
    while len(constraints) > checked_constraints:
      new_constraints = constraints[checked_constraints:]
      checked_constraints = len(constraints)
      for new_constraint in new_constraints:
        if not new_constraint.expression.penalty_expression.is_differentiable:
          raise ValueError("non-differentiable losses (e.g. the zero-one loss) "
                           "cannot be optimized--they can only be constrained")
        penalty_value = new_constraint.expression.penalty_expression.evaluate(
            self._structure_memoizer)
        constraint_value = new_constraint.expression.constraint_expression.evaluate(
            self._structure_memoizer)
        self._proxy_constraints.append(penalty_value)
        self._constraints.append(constraint_value)
        inputs += penalty_value.inputs
        inputs += constraint_value.inputs
        variables += penalty_value.variables
        variables += constraint_value.variables
        constraints += new_constraint.expression.extra_constraints

    # Extract the list of all input `Tensor`-like objects (or nullary functions
    # returning such).
    self._inputs = inputs.list

    # Explicitly create all of the variables. This also functions as a sanity
    # check: before this point, no variable should have been accessed
    # directly, and since their storage didn't exist yet, they couldn't have
    # been.
    self._variables = variables.list
    for variable in self._variables:
      variable.create(self._structure_memoizer)

  def objective(self):
    """Returns the objective function.

    Returns:
      A scalar `Tensor` that should be minimized.
    """
    return self._objective(self._structure_memoizer, {})

  @property
  def num_constraints(self):
    """Returns the number of constraints.

    Returns:
      An int containing the number of constraints.
    """
    num_constraints = len(self._constraints)
    assert num_constraints == len(self._proxy_constraints)
    return num_constraints

  def constraints(self):
    """Returns the `Tensor` of constraint functions.

    The constraints are element-wise non-positivity constraints, i.e. if g_i is
    the ith element of the constraints `Tensor`, then the ith constraint is g_i
    <= 0.

    Returns:
      A rank-1 `Tensor` of constraint functions.
    """
    value_memoizer = {}
    return tf.stack([
        cc(self._structure_memoizer, value_memoizer) for cc in self._constraints
    ])

  def proxy_constraints(self):
    """Returns the optional `Tensor` of proxy constraint functions.

    The difference between `constraints` and `proxy_constraints` is that the
    former are used when updating the Lagrange multipliers (or equivalent),
    while the latter are used for the model update.

    Returns:
      A rank-1 `Tensor` of proxy constraint functions.
    """
    value_memoizer = {}
    return tf.stack([
        cc(self._structure_memoizer, value_memoizer)
        for cc in self._proxy_constraints
    ])

  def components(self):
    """Returns all three of the objective, constraints and proxy constraints.

    Returns:
      A tuple of three `Tensors`, analagous to the results of objective(),
      constraints() and proxy_constraints(), respectively. If there are no proxy
      constraints, then the third component should be `None`.
    """
    # We use the same value_memoizer for each component, ensuring that the model
    # will be evaluated (differentiated) only once.
    value_memoizer = {}
    objective = self._objective(self._structure_memoizer, value_memoizer)
    constraints = tf.stack([
        cc(self._structure_memoizer, value_memoizer) for cc in self._constraints
    ])
    proxy_constraints = tf.stack([
        cc(self._structure_memoizer, value_memoizer)
        for cc in self._proxy_constraints
    ])
    return objective, constraints, proxy_constraints

  @property
  def inputs(self):
    """Returns a list of inputs to problem.

    Returns:
      A list of inputs, each of which is a `Tensor`-like object, or a nullary
      function returning such.
    """
    return self._inputs

  @property
  def variables(self):
    """Returns a list of variables owned by this problem.

    The returned variables will only be those that are owned by the rate
    minimization problem itself, e.g. implicit slack variables and denominator
    accumulators. The model variables will *not* be included.

    Returns:
      A list of variables.
    """
    # Variables do not have their values memoized, so there's no real need to
    # include a value_memoizer.
    return ([vv(self._structure_memoizer) for vv in self._variables] +
            [self._global_step])

  def update_ops(self):
    """Creates and returns a list of ops to run at the start of train_op.

    When a constrained optimizer creates a train_op, it will include these ops
    before the main training step. These ops update internal state that is used
    to estimate the denominators of rates.

    Returns:
      A list of ops.
    """
    value_memoizer = {}

    update_ops = []
    for variable in self._variables:
      update_ops += variable.update_ops(self._structure_memoizer,
                                        value_memoizer)

    # Increment our internal global_step after all of the other update_ops.
    with tf.control_dependencies(update_ops):
      update_ops = [self._global_step.assign_add(1)]

    return update_ops
