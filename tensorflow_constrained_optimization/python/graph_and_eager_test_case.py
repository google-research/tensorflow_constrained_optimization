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
      raise ValueError("_GraphWrappedSession.run expects a Tensor argument")
    return self._session.run(tensor)

  def run_ops(self, ops):
    """Executes the given collection of `Operation`s.

    This, like the `run()` method, is a special-case of `tf.Session.run()`.
    Unlike the `run()` method, it is intended to be used to execute
    `tf.Operation`s, not evaluate a `Tensor`.

    Args:
      ops: a collection of `Operation`s to execute.
    """
    self._session.run(ops)


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
      raise ValueError("_EagerWrappedSession.run expects a Tensor argument")
    return tensor.numpy()

  def run_ops(self, ops):
    """Ignores the given collection of `Operation`s.

    This, like the `run()` method, is a special-case of `tf.Session.run()`.
    Unlike the `run()` method, it is intended to be used to execute
    `tf.Operation`s, not evaluate a `Tensor`. In eager mode, operations are
    executed when they're created, so this method is a no-op.

    Args:
      ops: a collection of `Operation`s to execute.
    """
    del ops


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
