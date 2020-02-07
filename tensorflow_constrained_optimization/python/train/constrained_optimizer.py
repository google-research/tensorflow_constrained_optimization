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
"""Defines constrained optimizer base classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf

from tensorflow_constrained_optimization.python import constrained_minimization_problem


@six.add_metaclass(abc.ABCMeta)
class Formulation(object):
  """Represents a constrained optimization formulation.

  Currently, the two formulations that this library supports are the Lagrangian
  formulation, and the proxy-Lagrangian formulation. Both formulations have an
  associated internal state. For example, the Lagrangian formulation maintains a
  `Tensor` of Lagrange multipliers.

  Implementations of this class are responsible for maintaining the internal
  state (the "state" and "create_state" methods), and constructing the
  appropriate updates for both this state, and the parameters of the model that
  we're training (the "get_loss_fn" method).
  """

  @abc.abstractproperty
  def state(self):
    """Fetches the internal state, or returns None if it hasn't been created.

    Returns:
      The state `Tensor`, if create_state() has been called, or None otherwise.
    """

  @abc.abstractmethod
  def create_state(self, num_constraints):
    """Fetches the internal state, creating it if necessary.

    Args:
      num_constraints: int, the number of constraints in the
        `ConstrainedMinimizationProblem` that will eventually be minimized.

    Returns:
      The state `Tensor`.
    """

  @abc.abstractmethod
  def get_loss_fn(self, minimization_problem):
    """Returns the loss function.

    The resulting loss function should use `tf.custom_gradient` to override its
    gradients. First, the gradients w.r.t. the internal state should be written
    in terms of the constraints, instead of the proxy_constraints. Second, the
    gradients may be negated, depending on the formulation (for example, for the
    Lagrangian formulation, we wish to maximize over the Lagrange multipliers,
    so the associated gradients will be negated).

    Args:
      minimization_problem: `ConstrainedMinimizationProblem`, the problem to
        minimize.

    Returns:
      The loss function.
    """


def create_loss(formulation, minimization_problem):
  """Creates a loss function from a `ConstrainedMinimizationProblem`.

  Minimizing the returned loss will have the effect of jointly optimizing over
  both the internal state used by the given formulation, and the parameters of
  the model that we're training.

  In addition to a loss function, this method returns a function returning a
  list of operations that should be executed before each iteration
  ("update_ops"), and a `tf.Variable` containing the formulation's internal
  state.

  In graph mode, the result of this update_ops function could be "attached" to
  the train_op using tf.control_dependencies. In eager mode, it should be called
  before each iteration. Likewise, you should make sure to differentiate w.r.t.
  the internal state variable by e.g. including it in the var_list that you pass
  to an `Optimizer`'s minimize() method.

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
    formulation: `Formulation` (defined above) to use for performing constrained
      optimization.
    minimization_problem: `ConstrainedMinimizationProblem`, the problem to
      optimize.

  Returns:
    A (loss_fn, update_ops_fn, state_variable) tuple, where loss_fn is a nullary
    function returning a `Tensor` that can be minimized to optimize the
    constrained problem, update_ops_fn is a nullary function that returns a list
    of operations that should be executed before each training iteration, and
    multipliers_variable is a `tf.Variable` containing the internal state of the
    given formulation.
  """
  state_variable = formulation.create_state(
      minimization_problem.num_constraints)
  loss_fn = formulation.get_loss_fn(minimization_problem)

  return loss_fn, minimization_problem.update_ops, state_variable


class ConstrainedOptimizerV1(tf.compat.v1.train.Optimizer):
  """Base class representing a constrained V1 optimizer.

  A `ConstrainedOptimizerV1` wraps one or two `tf.compat.v1.train.Optimizer`s,
  and applies them to a `ConstrainedMinimizationProblem`. Like a
  `tf.compat.v1.train.Optimizer`, its minimize() method can be used to minimize
  a `Tensor` argument. Unlike a normal `tf.compat.v1.train.Optimizer`, however,
  a `ConstrainedOptimizerV1` can *instead* take a
  `ConstrainedMinimizationProblem` as the first parameter to minimize(), in
  which case it will perform constrained optimization.

  A `ConstrainedOptimizerV1` wraps a normal `tf.compat.v1.train.Optimizer` (the
  "optimizer" constructor parameter). If you minimize a `Tensor`, then the
  `ConstrainedOptimizerV1` will basically be an overly-complicated wrapper
  around this optimizer. The "constraint_optimizer" constructor parameter is
  used only for constrained optimization (i.e. when minimize() is given a
  `ConstrainedMinimizationProblem`), and is used to impose the constraints by
  maximizing over Lagrange multipliers (or their equivalents, if not using the
  Lagrangian formulation). If "constraint_optimizer" is not provided, then we
  will default to using the "optimizer" for the constraint portion of the
  optimization.

  All `ConstrainedOptimizerV1` implementations are stateful. The internal state
  can be accessed via the "trainable_variables" method, which should be included
  in the var_list parameter that you pass to "minimize" (unless var_list is
  None). *However*, you should be aware that the state will generally not exist
  before the first call to "minimize". The reason for this is that if we're
  using e.g. the Lagrangian formulation, then we need to know how many
  constraints we'll have before we can create the Lagrange multipliers. Hence,
  you may need to create the internal state explicitly, by providing the
  num_constraints argument to the constructor.
  """

  def __init__(self,
               formulation,
               optimizer,
               num_constraints=None,
               constraint_optimizer=None,
               name="ConstrainedOptimizerV1"):
    """Constructs a new `ConstrainedOptimizerV1`.

    Args:
      formulation: `Formulation` (defined above) to use for performing
        constrained optimization.
      optimizer: `tf.compat.v1.train.Optimizer`, used to optimize the objective
        and proxy_constraints portion of `ConstrainedMinimizationProblem`. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multipliers (or their analogues).
      num_constraints: optional int, the number of constraints in the
        `ConstrainedMinimizationProblem` that will eventually be minimized. If
        this argument is provided, then the internal state will be created
        inside this constructor. Otherwise, it will be created inside the first
        call to get_loss_fn().
      constraint_optimizer: optional `tf.compat.v1.train.Optimizer`, used to
        optimize the Lagrange multipliers (or their analogues).
      name: a non-empty string, which will be passed on to the parent
        `tf.compat.v1.train.Optimizer`'s constructor.

    Raises:
      TypeError: if optimizer or constraint_optimizer are implementations of
        `tf.keras.optimizers.Optimizer` instead of
        `tf.compat.v1.train.Optimizer`.
    """
    # The use_locking parameter does nothing here (the locking behavior will be
    # based on the use_locking parameters that were passed to the constructor(s)
    # of the wrapped optimizer and constraint_optimizer).
    super(ConstrainedOptimizerV1, self).__init__(use_locking=False, name=name)

    # Instead of checking that optimizer and (optionally) constraint_optimizer
    # are instances of tf.compat.v1.train.Optimizer, we check that they are
    # *not* instances of tf.keras.optimizers.Optimizer. The reason for this is
    # that we want to support duck typing, but at the same time, want to provide
    # the user with an early error message if what they're trying to do is
    # definitely wrong.
    if (isinstance(optimizer, tf.keras.optimizers.Optimizer) or
        isinstance(constraint_optimizer, tf.keras.optimizers.Optimizer)):
      raise TypeError("a V1 constrained optimizer must be constructed from a "
                      "V1 optimizer and (optionally) constraint_optimizer "
                      "(i.e. implementations of tf.compat.v1.train.Optimizer)")

    self._formulation = formulation
    self._optimizer = optimizer
    self._constraint_optimizer = constraint_optimizer

    if num_constraints is not None:
      self._formulation.create_state(num_constraints)

  # As in Optimizer, this is *not* a property.
  def variables(self):
    """Returns a list of variables owned by this optimizer.

    The returned variables will only be those that are owned by the constrained
    optimizer itself, or transitively by objects that it owns. These include the
    variables owned by the wrapped optimizer and constraint_optimizer, and the
    constrained formulation's internal state variable (e.g. Lagrange
    multipliers, for the Lagrangian formulation).

    If you didn't pass num_constraints to the constructor, the constrained
    formulation's internal state variables won't exist until the first call to
    minimize() or compute_gradients(), since we need to know the number of
    constraints in order to create them. For this reason, until you've started
    optimization, this method will return the empty list.

    Returns:
      A list of variables.
    """
    result = self._optimizer.variables()
    if self._constraint_optimizer is not None:
      result += self._constraint_optimizer.variables()
    if self._formulation.state is not None:
      result.append(self._formulation.state)
    return result

  def trainable_variables(self):
    """Returns a list of trainable variables owned by this optimizer.

    The returned variables will only be those that are owned by the constrained
    optimizer itself, or transitively by objects that it owns. These include the
    variables owned by the wrapped optimizer and constraint_optimizer, and the
    constrained formulation's internal state variable (e.g. Lagrange
    multipliers, for the Lagrangian formulation).

    If you didn't pass num_constraints to the constructor, the constrained
    formulation's internal state variables won't exist until the first call to
    minimize() or compute_gradients(), since we need to know the number of
    constraints in order to create them. For this reason, until you've started
    optimization, this method will return the empty list.

    Returns:
      A list of variables.
    """
    return [variable for variable in self.variables() if variable.trainable]

  def non_trainable_variables(self):
    """Returns a list of non-trainable variables owned by this optimizer.

    The returned variables will only be those that are owned by the constrained
    optimizer itself, or transitively by objects that it owns. These include the
    variables owned by the wrapped optimizer and constraint_optimizer.

    Returns:
      A list of variables.
    """
    return [variable for variable in self.variables() if not variable.trainable]

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients of a `ConstrainedMinimizationProblem` (or loss).

    If "loss" is a `ConstrainedMinimizationProblem` (which is the most common
    use-case for this method), *and* var_list is `None`, then the returned list
    of (gradient, variable) pairs will include an entry for the internal state
    variable of the constrained optimization formulation (e.g. the Lagrange
    multipliers, for the Lagrangian formulation). If var_list is not `None`,
    then these extra gradients will only be computed if the trainable_variables
    for this `ConstrainedOptimizerV1` are included in var_list.

    There is a complication here, however: the internal state won't exist until
    the first time we enter this function, because we cannot create it until we
    know the number of constraints). You can work around this by providing the
    num_constraints argument to the constructor.

    Inside apply_gradients(), the state gradient will be dispatched to the
    appropriate contained optimizer (i.e. either the "optimizer" or
    "constrained_optimizer"), and optionally projected, in the
    {,_resource}_apply_{dense,sparse}() methods.

    If "loss" is *not* a `ConstrainedMinimizationProblem`, then this function
    will thunk down to the compute_gradients() method of the contained
    "optimizer".

    Args:
      loss: either a `ConstrainedMinimizationProblem`, or, if we do not wish to
        perform constrained optimization, a loss `Tensor` (in graph mode) or a
        nullary function returning a loss `Tensor` (in eager mode). In the two
        latter cases, this function will thunk down to the compute_gradients()
        method of the contained "optimizer".
      var_list: as in `tf.compat.v1.train.Optimizer`.
      gate_gradients: as in `tf.compat.v1.train.Optimizer`.
      aggregation_method: as in `tf.compat.v1.train.Optimizer`.
      colocate_gradients_with_ops: as in `tf.compat.v1.train.Optimizer`.
      grad_loss: as in `tf.compat.v1.train.Optimizer`.

    Returns:
      A list of (gradient, variable) pairs, as in the compute_gradients() method
      of `tf.compat.v1.train.Optimizer`.
    """
    if not isinstance(
        loss, constrained_minimization_problem.ConstrainedMinimizationProblem):
      return super(ConstrainedOptimizerV1, self).compute_gradients(
          loss,
          var_list=var_list,
          gate_gradients=gate_gradients,
          aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops,
          grad_loss=grad_loss)

    if grad_loss is not None:
      raise ValueError("the grad_loss argument cannot be provided when the "
                       "loss argument is a ConstrainedMinimizationProblem")

    with tf.control_dependencies(loss.update_ops()):
      loss = self._formulation.get_loss_fn(loss)
      if not tf.executing_eagerly():
        loss = loss()
      return super(ConstrainedOptimizerV1, self).compute_gradients(
          loss,
          var_list=var_list,
          gate_gradients=gate_gradients,
          aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops,
          grad_loss=grad_loss)

  # pylint: disable=protected-access

  def _create_slots(self, var_list):
    state = self._formulation.state
    if self._constraint_optimizer is None or state is None:
      return self._optimizer._create_slots(var_list)

    state_var_list = []
    non_state_var_list = []
    for var in var_list:
      # We have to use "is" for the state comparison because in eager mode in
      # TensorFlow 2.1+, __eq__ is an element-wise comparison.
      if var is state:
        state_var_list.append(var)
      else:
        non_state_var_list.append(var)

    self._optimizer._create_slots(non_state_var_list)
    self._constraint_optimizer._create_slots(state_var_list)

  def _prepare(self):
    self._optimizer._prepare()
    if self._constraint_optimizer is not None:
      self._constraint_optimizer._prepare()

  def _apply_dense(self, gradient, variable, *args, **kwargs):
    assert variable is not None
    if (self._constraint_optimizer is not None and
        variable is self._formulation.state):
      return self._constraint_optimizer._apply_dense(gradient, variable, *args,
                                                     **kwargs)
    return self._optimizer._apply_dense(gradient, variable, *args, **kwargs)

  def _apply_sparse(self, gradient, variable, *args, **kwargs):
    assert variable is not None
    if (self._constraint_optimizer is not None and
        variable is self._formulation.state):
      return self._constraint_optimizer._apply_sparse(gradient, variable, *args,
                                                      **kwargs)
    return self._optimizer._apply_sparse(gradient, variable, *args, **kwargs)

  def _resource_apply_dense(self, gradient, handle, *args, **kwargs):
    assert handle is not None
    # TODO: is it safe to compare variables and handles? It works in
    # the tests, but will it *always* work?
    if (self._constraint_optimizer is not None and
        handle is self._formulation.state):
      return self._constraint_optimizer._resource_apply_dense(
          gradient, handle, *args, **kwargs)
    return self._optimizer._resource_apply_dense(gradient, handle, *args,
                                                 **kwargs)

  def _resource_apply_sparse(self, gradient, handle, *args, **kwargs):
    assert handle is not None
    # TODO: is it safe to compare variables and handles? It works in
    # the tests, but will it *always* work?
    if (self._constraint_optimizer is not None and
        handle is self._formulation.state):
      return self._constraint_optimizer._resource_apply_sparse(
          gradient, handle, *args, **kwargs)
    return self._optimizer._resource_apply_sparse(gradient, handle, *args,
                                                  **kwargs)

  # pylint: enable=protected-access


# FUTURE WORK: should we override get_gradients()?
class ConstrainedOptimizerV2(tf.keras.optimizers.Optimizer):
  """Base class representing a constrained V2 optimizer.

  A `ConstrainedOptimizerV2` wraps one or two `tf.keras.optimizers.Optimizer`s,
  and applies them to a `ConstrainedMinimizationProblem`. Like a
  `tf.keras.optimizers.Optimizer`, its minimize() method can be used to minimize
  a loss argument. Unlike a normal `tf.keras.optimizers.Optimizer`, however, a
  `ConstrainedOptimizerV2` can *instead* take a `ConstrainedMinimizationProblem`
  as the first parameter to minimize(), in which case it will perform
  constrained optimization.

  A `ConstrainedOptimizerV2` wraps a normal `tf.keras.optimizers.Optimizer` (the
  "optimizer" constructor parameter). If you minimize a loss, then the
  `ConstrainedOptimizerV2` will basically be an overly-complicated wrapper
  around this optimizer. The "constraint_optimizer" constructor parameter is
  used only for constrained optimization (i.e. when minimize() is given a
  `ConstrainedMinimizationProblem`), and is used to impose the constraints by
  maximizing over Lagrange multipliers (or their equivalents, if not using the
  Lagrangian formulation). If "constraint_optimizer" is not provided, then we
  will default to using the "optimizer" for the constraint portion of the
  optimization.

  All `ConstrainedOptimizerV2` implementations are stateful. The internal state
  can be accessed via the "trainable_variables" method, which should be included
  in the var_list parameter that you pass to "minimize".
  """

  def __init__(self,
               formulation,
               optimizer,
               num_constraints,
               constraint_optimizer=None,
               name="ConstrainedOptimizerV2"):
    """Constructs a new `ConstrainedOptimizerV2`.

    Args:
      formulation: `Formulation` (defined above) to use for performing
        constrained optimization.
      optimizer: `tf.keras.optimizers.Optimizer`, used to optimize the objective
        and proxy_constraints portion of `ConstrainedMinimizationProblem`. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multipliers (or their analogues).
      num_constraints: int, the number of constraints in the
        `ConstrainedMinimizationProblem` that will eventually be minimized.
      constraint_optimizer: optional `tf.keras.optimizers.Optimizer`, used to
        optimize the Lagrange multipliers (or their analogues).
      name: a non-empty string, which will be passed on to the parent
        `tf.keras.optimizers.Optimizer`'s  constructor.

    Raises:
      TypeError: if optimizer or constraint_optimizer are implementations of
        `tf.compat.v1.train.Optimizer` instead of
        `tf.keras.optimizers.Optimizer`.
    """
    super(ConstrainedOptimizerV2, self).__init__(name=name)

    # Instead of checking that optimizer and (optionally) constraint_optimizer
    # are instances of tf.keras.optimizers.Optimizer, we check that they are
    # *not* instances of tf.compat.v1.train.Optimizer. The reason for this is
    # that we want to support duck typing, but at the same time, want to provide
    # the user with an early error message if what they're trying to do is
    # definitely wrong.
    if (isinstance(optimizer, tf.compat.v1.train.Optimizer) or
        isinstance(constraint_optimizer, tf.compat.v1.train.Optimizer)):
      raise TypeError("a V2 constrained optimizer must be constructed from a "
                      "V2 optimizer and (optionally) constraint_optimizer "
                      "(i.e. implementations of tf.keras.optimizers.Optimizer)")

    self._formulation = formulation
    self._optimizer = optimizer
    self._constraint_optimizer = constraint_optimizer

    self._formulation.create_state(num_constraints)

  # As in Optimizer, this is *not* a property.
  def variables(self):
    """Returns a list of variables owned by this optimizer.

    The returned variables will only be those that are owned by the constrained
    optimizer itself, or transitively by objects that it owns. These include the
    variables owned by the wrapped optimizer and constraint_optimizer, and the
    constrained formulation's internal state variable (e.g. Lagrange
    multipliers, for the Lagrangian formulation).

    Returns:
      A list of variables.
    """
    result = self._optimizer.variables()
    if self._constraint_optimizer is not None:
      result += self._constraint_optimizer.variables()
    if self._formulation.state is not None:
      result.append(self._formulation.state)
    return result

  def trainable_variables(self):
    """Returns a list of trainable variables owned by this optimizer.

    The returned variables will only be those that are owned by the constrained
    optimizer itself, or transitively by objects that it owns. These include the
    variables owned by the wrapped optimizer and constraint_optimizer, and the
    constrained formulation's internal state variable (e.g. Lagrange
    multipliers, for the Lagrangian formulation).

    Returns:
      A list of variables.
    """
    return [variable for variable in self.variables() if variable.trainable]

  def non_trainable_variables(self):
    """Returns a list of non-trainable variables owned by this optimizer.

    The returned variables will only be those that are owned by the constrained
    optimizer itself, or transitively by objects that it owns. These include the
    variables owned by the wrapped optimizer and constraint_optimizer.

    Returns:
      A list of variables.
    """
    return [variable for variable in self.variables() if not variable.trainable]

  def _split_var_list(self, var_list):
    """Helper function that splits a var_list between the two optimizers."""
    # This assertion cannot fail, since we only call this method after checking
    # that constraint_optimizer is non-None.
    assert self._constraint_optimizer is not None

    state = self._formulation.state
    if state is None:
      return var_list, []

    state_var_list = []
    non_state_var_list = []
    for var in var_list:
      # We have to use "is" for the state comparison because in eager mode in
      # TensorFlow 2.1+, __eq__ is an element-wise comparison.
      if var is state:
        state_var_list.append(var)
      else:
        non_state_var_list.append(var)

    return non_state_var_list, state_var_list

  # pylint: disable=protected-access

  def _compute_gradients(self, loss, var_list, grad_loss=None):
    """Compute gradients of a `ConstrainedMinimizationProblem` (or loss).

    If "loss" is a `ConstrainedMinimizationProblem` (which is the most common
    use-case for this method), then you'll want to make sure that var_list
    includes the internal state variable of the constrained optimization
    formulation (e.g. the Lagrange multipliers, for the Lagrangian formulation).

    Inside apply_gradients(), the state gradient will be dispatched to the
    appropriate contained optimizer (i.e. either the "optimizer" or
    "constrained_optimizer"), and optionally projected, in the
    _resource_apply_{dense,sparse}() methods.

    If "loss" is *not* a `ConstrainedMinimizationProblem`, then this function
    will thunk down to the _compute_gradients() method of the contained
    "optimizer".

    Args:
      loss: either a `ConstrainedMinimizationProblem`, or, if we do not wish to
        perform constrained optimization, a loss `Tensor` (in graph mode) or a
        nullary function returning a loss `Tensor` (in eager mode). In the two
        latter cases, this function will thunk down to the compute_gradients()
        method of the contained "optimizer".
      var_list: as in `tf.keras.optimizers.Optimizer`.
      grad_loss: as in `tf.keras.optimizers.Optimizer`.

    Returns:
      A list of (gradient, variable) pairs, as in the _compute_gradients()
      method of `tf.keras.optimizers.Optimizer`.
    """
    if not isinstance(
        loss, constrained_minimization_problem.ConstrainedMinimizationProblem):
      return super(ConstrainedOptimizerV2, self)._compute_gradients(
          loss, var_list=var_list, grad_loss=grad_loss)

    if grad_loss is not None:
      raise ValueError("the grad_loss argument cannot be provided when the "
                       "loss argument is a ConstrainedMinimizationProblem")

    with tf.control_dependencies(loss.update_ops()):
      loss_fn = self._formulation.get_loss_fn(loss)
      return super(ConstrainedOptimizerV2, self)._compute_gradients(
          loss_fn, var_list=var_list, grad_loss=grad_loss)

  def _create_slots(self, var_list):
    if self._constraint_optimizer is None:
      return self._optimizer._create_slots(var_list)

    var_list, state_var_list = self._split_var_list(var_list)
    self._optimizer._create_slots(var_list)
    self._constraint_optimizer._create_slots(state_var_list)

  def _prepare(self, var_list):
    if self._constraint_optimizer is None:
      return self._optimizer._prepare(var_list)

    var_list, state_var_list = self._split_var_list(var_list)
    self._optimizer._prepare(var_list)
    self._constraint_optimizer._prepare(state_var_list)

  def _create_hypers(self):
    self._optimizer._create_hypers()
    if self._constraint_optimizer is not None:
      self._constraint_optimizer._create_hypers()

  def get_config(self):
    # The problem here is that, while we can easily serialize hyperparameters of
    # the contained Optimizers, we cannot easily serialize their *types*.
    # FUTURE WORK: find a way to implement this method.
    raise NotImplementedError("ConstrainedOptimizerV2s cannot be serialized")

  def _resource_apply_dense(self, gradient, handle, *args, **kwargs):
    assert handle is not None
    # TODO: is it safe to compare variables and handles? It works in
    # the tests, but will it *always* work?
    if (self._constraint_optimizer is not None and
        handle is self._formulation.state):
      return self._constraint_optimizer._resource_apply_dense(
          gradient, handle, *args, **kwargs)
    return self._optimizer._resource_apply_dense(gradient, handle, *args,
                                                 **kwargs)

  def _resource_apply_sparse(self, gradient, handle, *args, **kwargs):
    assert handle is not None
    # TODO: is it safe to compare variables and handles? It works in
    # the tests, but will it *always* work?
    if (self._constraint_optimizer is not None and
        handle is self._formulation.state):
      return self._constraint_optimizer._resource_apply_sparse(
          gradient, handle, *args, **kwargs)
    return self._optimizer._resource_apply_sparse(gradient, handle, *args,
                                                  **kwargs)

  # pylint: enable=protected-access
