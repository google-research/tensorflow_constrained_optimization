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

import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python import constrained_minimization_problem


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
    # We make a fake 1-parameter linear objective so that we don't get a "no
    # variables to optimize" error.
    self._objective = tf.Variable(0.0, trainable=True, dtype=tf.float32)
    self._constraints = tf.constant(constraints, dtype=tf.float32)

  def objective(self):
    """Returns the objective function."""
    return self._objective

  @property
  def num_constraints(self):
    constraints_dims = self._constraints.shape.dims
    assert constraints_dims is not None
    assert all([dim.value is not None for dim in constraints_dims])

    if not constraints_dims:
      return 0

    size = 1
    for ii in constraints_dims:
      size *= ii.value
    return int(size)

  def constraints(self):
    """Returns the constant constraint violations."""
    return self._constraints

  @property
  def variables(self):
    """Returns a list of variables owned by this problem."""
    return [self._objective]
