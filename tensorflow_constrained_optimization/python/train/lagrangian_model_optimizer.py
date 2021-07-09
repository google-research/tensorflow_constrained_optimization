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
interface. However, it has an extra wrinkle: the Lagrange multipliers are not
necessarily a `Tensor` of numbers, but instead are provided by the user in the
form of a *model*. This enables one to have feature-dependent constraints, with
corresponding feature-dependent Lagrange multiplier models.

For more specifics, please refer to:

> Narasimhan, Cotter, Zhou, Wang and Guo. "Approximate Heavily-Constrained
> Learning with Lagrange Multiplier Models". NeurIPS'20 (to appear)

Example
=======

Suppose that we already have a `ConstrainedMinimizationProblem` named "problem"
that we wish to optimize. The create_lagrangian_model_loss() function takes a
"create_fn" parameter that, when called, expects to be passed the number of
constraints, and will return a `tf.Module` containing the Lagrange multiplier
model. This model is itself callable as a nullary function, which evaluates the
model and returns the Lagrange multipliers.

The simplest way to accomplish this would be to think of "create_fn" as a
*constructor*. For example, if we just want to use the usual Lagrangian
formulation, we could create the following class:

```python
class Lagrangian(tf.Module):

  def __init__(self, num_constraints):
    super(Lagrangian, self).__init__()
    initial_lagrange_multipliers = np.zeros((num_constraints,),
                                            dtype=np.float32)
    self._lagrange_multipliers = tf.Variable(
        initial_lagrange_multipliers, trainable=True, dtype=tf.float32)

  def __call__(self):
    return self._lagrange_multipliers
```

Then, we could create a loss by calling:

```python
loss_fn, update_ops_fn, trainable_variables = tfco.create_lagrangian_model_loss(
    minimization_problem=problem, create_fn=Lagrangian)
```

We can now minimize "loss_fn" using a normal TensorFlow optimizer, provided that
we ensure that "update_ops_fn" is called before every iteration. The
"trainable_variables" collection will contain the trainable parameters of the
Lagrange multiplier model (in this case, the Lagrange multipliers themselves).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_constrained_optimization.python.train import lagrangian_optimizer


class _LagrangianModelFormulation(lagrangian_optimizer._LagrangianFormulation):  # pylint: disable=protected-access
  """Contains the internal state for the Lagrangian-model formulation.

  To optimize using the Lagrangian-model formulation, we jointly minimize over
  the model parameters, and maximize over the Lagrange multiplier model.
  """

  def __init__(self, create_fn, dual_scale=1.0, variable_fn=tf.Variable):
    """Constructs a new `_LagrangianModelFormulation`.

    Args:
      create_fn: a function that takes a natural number as a parameter (the
        number of constraints), and returns a `tf.Module` instance containing
        the Lagrangian model's state, with `__call__(self)` implemented (i.e.
        it's callable as a nullary function that returns the values of the
        Lagrange multipliers). Typically, create_fn will be a *class* inheriting
        from `tf.Module`, so calling it will call the constructor, and return an
        instance.
      dual_scale: optional float defaulting to 1, a multiplicative scaling
        factor applied to gradients w.r.t. the Lagrange multipliers.
      variable_fn: optional function with the same signature as the
        `tf.Variable` constructor, that returns a new variable with the
        specified properties.

    Raises:
      ValueError: if "maximum_multiplier_radius" or "dual_scale" are
        non-positive.
    """
    super(_LagrangianModelFormulation, self).__init__(
        dual_scale=dual_scale, variable_fn=variable_fn)

    self._create_fn = create_fn
    self._evaluate_fn = None
    self._num_constraints = None

  def state(self):
    if self._evaluate_fn is None:
      raise RuntimeError("create_state must be called before state is "
                         "evaluated")

    # Lagrange multipliers must be nonnegative, so we take the absolute value.
    # This is better than clipping them at zero because we will still have
    # meaningful nonzero gradients for negative numbers. However, we can't use
    # tf.abs() since we always want to have a nonzero gradient, even at zero
    # (where tf.abs() has a zero gradient).
    lagrange_multipliers = self._evaluate_fn()
    return tf.where(lagrange_multipliers >= 0, lagrange_multipliers,
                    -lagrange_multipliers)

  def create_state(self, num_constraints):
    if num_constraints is None:
      raise ValueError("num_constraints cannot be None--did you mean 0?")
    elif self._num_constraints is None:
      self._evaluate_fn = self._create_fn(num_constraints)
      if not isinstance(self._evaluate_fn, tf.Module):
        raise RuntimeError("calling create_fn must result in a tf.Module")
      self._num_constraints = num_constraints
    elif self._num_constraints != num_constraints:
      raise ValueError("you cannot change the number of constraints from {} to "
                       "{} in the middle of optimization".format(
                           self._num_constraints, num_constraints))


def create_lagrangian_model_loss(minimization_problem,
                                 create_fn,
                                 dual_scale=1.0,
                                 variable_fn=tf.Variable):
  """Creates a loss function from a `ConstrainedMinimizationProblem`.

  Args:
    minimization_problem: `ConstrainedMinimizationProblem`, the problem to
      optimize.
    create_fn: a function that takes a natural number as a parameter (the number
      of constraints), and returns a `tf.Module` instance containing the
      Lagrangian model's state, with `__call__(self)` implemented (i.e. it's
      callable as a nullary function that returns the values of the Lagrange
      multipliers). Typically, create_fn will be a *class* inheriting from
      `tf.Module`, so calling it will call the constructor, and return an
      instance.
    dual_scale: optional float defaulting to 1, a multiplicative scaling factor
      applied to gradients w.r.t. the Lagrange multipliers.
    variable_fn: optional function with the same signature as the `tf.Variable`
      constructor, that returns a new variable with the specified properties.

  Returns:
    A (loss_fn, update_ops_fn, trainable_variables) tuple, where loss_fn is a
    nullary function returning a `Tensor` that can be minimized to optimize the
    constrained problem, update_ops_fn is a nullary function that returns a list
    of operations that should be executed before each training iteration, and
    trainable_variables is a collection of parameters to the Lagrange multiplier
    model.
  """
  formulation = _LagrangianModelFormulation(
      create_fn=create_fn, dual_scale=dual_scale, variable_fn=variable_fn)
  loss_fn = formulation.get_loss_fn(minimization_problem)
  return (loss_fn, minimization_problem.update_ops,
          formulation.trainable_variables)
