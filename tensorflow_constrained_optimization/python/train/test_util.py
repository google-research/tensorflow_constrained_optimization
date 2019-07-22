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
"""Contains helpers used by tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_constrained_optimization.python import constrained_minimization_problem


class _GraphWrappedSession(object):
  """A simple `tf.Session` wrapper for graph mode."""

  def __init__(self, session):
    """Wraps a `_GraphWrappedSession` around the given `Session`."""
    if tf.executing_eagerly():
      raise ValueError("_GraphWrappedSession should only be used in graph mode")
    self._outer_session = session

  def __enter__(self):
    """Enters the session scope.

    Unlike the usual __enter__ method for a `Session`, this method also
    initializes global variables. The reason for this is to hide another
    difference between graph and eager modes.

    Returns:
      This object.
    """
    self._session = self._outer_session.__enter__()
    self._session.run(tf.global_variables_initializer())
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the session scope."""
    return self._outer_session.__exit__(exc_type, exc_value, traceback)

  def run(self, tensor):
    """Evaluates the given `Tensor`.

    Unlike `tf.Session.run()`, this method expects a single `Tensor` argument
    (i.e. not a list of `Tensor`s or any other more complex structure).

    Args:
      tensor: the `Tensor` to evaluate.

    Returns:
      The value of the given `Tensor`.

    Raises:
      ValueError: if the given tensor is not a `Tensor`.
    """
    if not tf.is_tensor(tensor):
      raise ValueError("_GraphWrappedSession.run must expects a Tensor "
                       "argument")
    return self._session.run(tensor)


class _EagerWrappedSession(object):
  """A simple `tf.Session` wrapper for eager mode."""

  def __init__(self):
    """Creates a new `_EagerWrappedSession`."""
    if not tf.executing_eagerly():
      raise ValueError("_EagerWrappedSession should only be used in eager mode")

  def __enter__(self):
    """Enters the session scope.

    Returns:
      This object.
    """
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the session scope."""
    del exc_type, exc_value, traceback

  def run(self, tensor):
    """Evaluates the given `Tensor`.

    Unlike `tf.Session.run()`, this method expects a single `Tensor` argument
    (i.e. not a list of `Tensor`s or any other more complex structure).

    Args:
      tensor: the `Tensor` to evaluate.

    Returns:
      The value of the given `Tensor`.

    Raises:
      ValueError: if the given tensor is not a `Tensor`.
    """
    if not tf.is_tensor(tensor):
      raise ValueError("_EagerWrappedSession.run must expects a Tensor "
                       "argument")
    return tensor.numpy()


class GraphAndEagerTestCase(tf.test.TestCase):
  """A `TestCase` including the `wrapped_session()` method."""

  def wrapped_session(self):
    """Returns a wrapper object around a session.

    In graph mode, this wrapper does nothing but slightly simplify the interface
    to an underlying `tf.Session`. In eager mode, there is no underlying
    `tf.Session` at all, but the same interface is implemented.

    Returns:
      A `_GraphWrappedSession` or `_EagerWrappedSession`, depending on whether
      we're executing in graph or eager mode.
    """
    if tf.executing_eagerly():
      return _EagerWrappedSession()
    return _GraphWrappedSession(self.session())


class ConstantMinimizationProblem(
    constrained_minimization_problem.ConstrainedMinimizationProblem):
  """A `ConstrainedMinimizationProblem` with constant constraint violations.

  This minimization problem is intended for use in performing simple tests of
  the Lagrange multiplier (or equivalent) update in the optimizers. There is a
  one-element "dummy" model parameter, but it should be ignored.
  """

  def __init__(self, constraints):
    """Constructs a new `ConstantMinimizationProblem`.

    Args:
      constraints: 1d numpy array, the constant constraint violations.

    Returns:
      A new `ConstantMinimizationProblem`.
    """
    # We make an fake 1-parameter linear objective so that we don't get a "no
    # variables to optimize" error.
    self._objective = tf.Variable(0.0, dtype=tf.float32)
    self._constraints = tf.constant(constraints, dtype=tf.float32)

  def objective(self):
    """Returns the objective function."""
    return self._objective

  @property
  def num_constraints(self):
    constraints_shape = self._constraints.shape.dims
    assert constraints_shape is not None
    assert all([dim.value is not None for dim in constraints_shape])

    if not constraints_shape:
      return 0

    size = 1
    for ii in constraints_shape:
      size *= ii.value
    return int(size)

  def constraints(self):
    """Returns the constant constraint violations."""
    return self._constraints
