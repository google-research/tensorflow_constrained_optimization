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
"""Defines abstract class for `ConstrainedMinimizationProblem`s.

In its simplest form, a `ConstrainedMinimizationProblem` consists of an
objective function to minimize, and a set of constraint functions that are
constrained to be non-positive, for example:

  minimize f(x)
  subject to g_i(x) <= 0 for i in {1,2,...,m}

where the functions f() and g_i() are represented as `Tensor`s (in graph mode)
or functions returning `Tensor`s (in eager mode).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class ConstrainedMinimizationProblem(object):
  """Abstract class representing a `ConstrainedMinimizationProblem`.

  A `ConstrainedMinimizationProblem` consists of an objective function to
  minimize, and a set of constraint functions that are constrained to be
  non-positive.

  In addition to the constraint functions, there may (optionally) be proxy
  constraint functions: a constrained optimizer will attempt to penalize these
  proxy constraint functions so as to satisfy the (non-proxy) constraints. Proxy
  constraints could be used if the constraints functions are difficult or
  impossible to optimize (e.g. if they're piecewise constant), in which case the
  proxy constraints should be some approximation of the original constraints
  that is well-enough behaved to permit successful optimization.
  """

  @abc.abstractmethod
  def objective(self):
    """Returns the objective function.

    In eager mode, this method should actually compute the objective function,
    instead of merely returning a cached `Tensor` as one might do in graph mode.

    Returns:
      A scalar `Tensor` that should be minimized.
    """

  @abc.abstractproperty
  def num_constraints(self):
    """Returns the number of constraints.

    The `Tensor`s returned by the constraints() and proxy_constraints() methods
    should each contain a total of num_constraints elements.

    Returns:
      An int containing the number of constraints.
    """

  @abc.abstractmethod
  def constraints(self):
    """Returns the `Tensor` of constraint functions.

    The constraints are element-wise non-positivity constraints, i.e. if g_i is
    the ith element of the constraints `Tensor`, then the ith constraint is g_i
    <= 0.

    In eager mode, this method should actually compute the constraint functions,
    instead of merely returning a cached `Tensor` as one might do in graph mode.

    Returns:
      A `Tensor` of constraint functions. If the proxy_constraints() method is
      defined, then both must return a `Tensor` with the same shape.
    """

  # This is a method, instead of an abstract method, since it doesn't need to be
  # overridden: if you don't override it, then there are no proxy constraints
  # (we'll just use the constraints).
  def proxy_constraints(self):
    """Returns the optional `Tensor` of proxy constraint functions.

    The difference between "constraints" and "proxy_constraints" is that, when
    proxy constraints are present, the "constraints" are merely EVALUATED during
    optimization, whereas the "proxy_constraints" are DIFFERENTIATED. If there
    are no proxy constraints, then the "constraints" are both evaluated and
    differentiated.

    For example, if we want to impose constraints on step functions, then we
    could use these functions for "constraints". However, because a step
    function has zero gradient almost everywhere, we can't differentiate these
    functions, so we would take "proxy_constraints" to be some differentiable
    approximation of "constraints".

    In eager mode, this method should actually compute the proxy constraint
    functions, instead of merely returning a cached `Tensor` as one might do in
    graph mode.

    Returns:
      A `Tensor` of proxy constraint functions with the same shape as that
      returned by the constraints() method, or `None` if the proxy constraints
      are identical to the constraints.
    """
    return

  def components(self):
    """Returns all three of the objective, constraints and proxy constraints.

    The default implementation of this method simply calls objective(),
    constraints() and proxy_constraints(), and returns the results in a tuple.
    In eager mode, however, this could be unnecessarily wasteful, since it could
    cause the model to be evaluated (and differentiated) three separate times,
    once for each component.

    To avoid this, users can override this method, and evaluate the model only
    once. They can then use the same cached evaluation as an input to each of
    the three components.

    Returns:
      A tuple of three `Tensors`, analagous to the results of objective(),
      constraints() and proxy_constraints(), respectively. If there are no proxy
      constraints, then the third component should be `None`.
    """
    return self.objective(), self.constraints(), self.proxy_constraints()

  # This is a method, instead of an abstract method, since it doesn't need to be
  # overridden: if variables returns [], then there are no Variables owned by
  # this problem.
  @property
  def variables(self):
    """Returns a list of variables owned by this problem.

    The returned variables will only be those that are owned by the constrained
    minimization problem itself, e.g. implicit slack variables. The model
    variables will *not* be included.

    Returns:
      A list of variables.
    """
    return []

  @property
  def trainable_variables(self):
    """Returns a list of trainable variables owned by this problem.

    The returned variables will only be those that are owned by the constrained
    minimization problem itself, e.g. implicit slack variables. The model
    variables will *not* be included.

    Returns:
      A list of variables.
    """
    return [variable for variable in self.variables if variable.trainable]

  @property
  def non_trainable_variables(self):
    """Returns a list of non-trainable variables owned by this problem.

    The returned variables will only be those that are owned by the constrained
    minimization problem itself, e.g. implicit slack variables. The model
    variables will *not* be included.

    Returns:
      A list of variables.
    """
    return [variable for variable in self.variables if not variable.trainable]

  # This is a method, instead of an abstract method, since it doesn't need to be
  # overridden: if update_ops() returns [], then there are no ops to run before
  # train_op.
  def update_ops(self):
    """Creates and returns a list of ops to run at the start of train_op.

    When a constrained optimizer calculates gradients or creates a train_op, it
    will include these ops before the main training step.

    In eager mode, this method should actually perform the required operations.
    In graph mode, it needs only to return the ops.

    Returns:
      A list of ops.
    """
    return []
