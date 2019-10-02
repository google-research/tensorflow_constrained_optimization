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
"""Contains a TestCase class with helpers supporting graph and eager modes.

A `GraphAndEagerTestCase` includes two methods in addition to those inherited
from `tf.test.TestCase`: wrapped_session() and wrapped_placeholder(). The former
is an alternative to the session() method: it results in a `Session`-like object
that supports both graph and eager modes. Similarly, wrapped_placeholder()
returns an object that can be treated similarly to a placeholder `Tensor`, but
can be used in both graph and eager modes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf


class _EagerPlaceholder(object):
  """A simple fake placeholder for eager mode.

  The interface of this object is a bit different than a normal placeholder,
  which, in graph mode, is simply a `Tensor`. An `_EagerPlaceholder` actually
  looks like a nullary function returning a `Tensor`.
  """

  def __init__(self):
    """Creates a new fake placeholder with value `None`."""
    if not tf.executing_eagerly():
      raise ValueError("_EagerPlaceholder should only be used in eager mode")
    self._tensor = None

  def set(self, tensor):
    """Assigns the value of this fake placeholder."""
    self._tensor = tensor

  def __call__(self):
    """Accesses the value of this fake placeholder."""
    return self._tensor


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
    self._session.run(tf.compat.v1.global_variables_initializer())
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the session scope."""
    return self._outer_session.__exit__(exc_type, exc_value, traceback)

  def run(self, tensor, feed_dict=None):
    """Evaluates the given `Tensor`.

    Unlike `tf.Session.run()`, this method expects a single `Tensor` argument
    (i.e. not a list of `Tensor`s or any other more complex structure).

    Args:
      tensor: the `Tensor` to evaluate, or a nullary function returning a
        `Tensor`.
      feed_dict: dict mapping placeholder `Tensor`s to their desired values.

    Returns:
      The value of the given `Tensor`.

    Raises:
      TypeError: if (i) the given tensor is neither a `Tensor` nor a nullary
        function returning a `Tensor`, or (ii) we're given a feed_dict but the
        tensor argument is not callable.
    """
    if callable(tensor):
      tensor = tensor()
    elif feed_dict:
      raise TypeError("if a feed_dict is provided to run(), then the tensor "
                      "argument must be a nullary function returning a Tensor")

    if not tf.is_tensor(tensor):
      raise TypeError("_GraphWrappedSession.run expects a Tensor argument, "
                      "or a nullary function returning a Tensor")

    return self._session.run(tensor, feed_dict=feed_dict)

  def run_ops(self, callback, feed_dict=None):
    """Executes the collection of ops returned by the given callback.

    This, like the `run()` method, is a special-case of `tf.Session.run()`.
    Unlike the `run()` method, it is used to execute ops, not evaluate a
    `Tensor`.

    Args:
      callback: a nullary function returning a collection of ops to execute.
      feed_dict: dict mapping placeholder `Tensor`s to their desired values.
    """
    self._session.run(callback(), feed_dict=feed_dict)


class _EagerWrappedSession(object):
  """A simple `tf.Session` wrapper look-alike for eager mode."""

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

  def run(self, tensor, feed_dict=None):
    """Evaluates the given `Tensor`.

    Unlike `tf.Session.run()`, this method expects a single `Tensor` argument
    (i.e. not a list of `Tensor`s or any other more complex structure).

    Args:
      tensor: the `Tensor` to evaluate, or a nullary function returning a
        `Tensor`. If the feed_dict argument is provided, then this *must* be a
        function, instead of a `Tensor`.
      feed_dict: dict mapping `_EagerPlaceholder`s to their desired values.

    Returns:
      The value of the given `Tensor`.

    Raises:
      TypeError: if (i) the given tensor is neither a `Tensor` nor a nullary
        function returning a `Tensor`, (ii) we're given a feed_dict but the
        tensor argument is not callable, or (iii) any of the feed_dict keys are
        not `_EagerPlaceholder`s.
    """
    if feed_dict:
      if not callable(tensor):
        raise TypeError("if a feed_dict is provided to run(), then the tensor "
                        "argument must be a nullary function returning a "
                        "Tensor")

      for key, value in six.iteritems(feed_dict):
        if not isinstance(key, _EagerPlaceholder):
          raise TypeError("in eager mode, feed_dict keys must be "
                          "_EagerPlaceholder objects")
        key.set(value)

      tensor = tensor()

      # Before returning, clear out the placeholders to make sure that the
      # *next* call to run() or run_ops() doesn't simply inherit their
      # placeholders from here.
      for key, value in six.iteritems(feed_dict):
        key.set(None)
    elif callable(tensor):
      tensor = tensor()

    if not tf.is_tensor(tensor):
      raise TypeError("_EagerWrappedSession.run expects a Tensor argument, "
                      "or a nullary function returning a Tensor")

    return tensor.numpy()

  def run_ops(self, callback, feed_dict=None):
    """Calls the given callback.

    This, like the `run()` method, is a special-case of `tf.Session.run()`.
    Unlike the `run()` method, it is used to execute ops, not evaluate a
    `Tensor`.

    Args:
      callback: a nullary function that performs the desired operations.
      feed_dict: dict mapping `_EagerPlaceholder`s to their desired values.

    Raises:
      TypeError: if any of the feed_dict keys are not `_EagerPlaceholder`s.
    """
    if feed_dict:
      for key, value in six.iteritems(feed_dict):
        if not isinstance(key, _EagerPlaceholder):
          raise TypeError("in eager mode, feed_dict keys must be "
                          "_EagerPlaceholder objects")
        key.set(value)

    # The main reason that run_ops() takes a callback, instead of a list of ops
    # (which would be ignored in eager mode), is to enable us to account for the
    # feed_dict *before* the callback is called.
    callback()

    # Before returning, clear out the placeholders to make sure that the *next*
    # call to run() or run_ops() doesn't simply inherit their placeholders from
    # here.
    if feed_dict:
      for key, value in six.iteritems(feed_dict):
        key.set(None)


class GraphAndEagerTestCase(tf.test.TestCase):
  """A `TestCase` including the `wrapped_session()` method."""

  def wrapped_placeholder(self, dtype, shape=None, name=None):
    """Returns a placeholder in graph mode, or a fake in eager mode."""
    if tf.executing_eagerly():
      return _EagerPlaceholder()
    return tf.compat.v1.placeholder(dtype, shape=shape, name=name)

  def wrapped_session(self):
    """Returns a wrapper object around a session.

    In graph mode, this wrapper does nothing but slightly simplify the interface
    to an underlying `tf.Session`. In eager mode, there is no underlying
    `tf.Session` at all, but a similar interface is implemented.

    Returns:
      A `_GraphWrappedSession` or `_EagerWrappedSession`, depending on whether
      we're executing in graph or eager mode.
    """
    if tf.executing_eagerly():
      return _EagerWrappedSession()
    return _GraphWrappedSession(self.session())
