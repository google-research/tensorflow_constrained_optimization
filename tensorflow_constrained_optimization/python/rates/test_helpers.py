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
"""Contains internal helpers for tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin


def _associate_functions_with_variables(size, evaluate_fn):
  """Finds a 1:1 correspondence between functions and variables.

  Suppose that we have n functions and n variables, and that each of the former
  is a function of *exactly* one of the latter. This situation commonly results
  from our use for thresholds and slack variables.

  This function will determine which functions depend on which variables.

  Args:
    size: int, the number of functions and variables.
    evaluate_fn: function mapping a numpy array of variable values to another
      numpy array of function values.

  Returns:
    A list of length "size", in which the ith element is the function that uses
    the ith variable. Equivalently, this list can be interpreted as a
    permutation that should be applied to the functions, in order to ensure that
    the ith variable is used by the ith function.

  Raises:
    RuntimeError: if we fail to successfully find the assignment. This is most
      likely to happen if there is not actually a 1:1 correspondence between
      functions and variables.
  """
  # FUTURE WORK: make these into (optional) arguments.
  default_value = 1000
  changed_value = -1000

  assignments = []
  for ii in xrange(size):
    assignments.append(set(xrange(size)))

  default_values = evaluate_fn([default_value] * size)

  num_loops = int(math.ceil(math.log(size, 2)))
  bit = 1
  for ii in xrange(num_loops):
    variable_indices1 = [ii for ii in xrange(size) if ii & bit != 0]
    variable_indices2 = [ii for ii in xrange(size) if ii & bit == 0]

    arguments = np.array([default_value] * size)
    arguments[variable_indices1] = changed_value
    values1 = evaluate_fn(arguments)
    function_indices1 = [
        ii for ii in xrange(size) if default_values[ii] != values1[ii]
    ]

    arguments = np.array([default_value] * size)
    arguments[variable_indices2] = changed_value
    values2 = evaluate_fn(arguments)
    function_indices2 = [
        ii for ii in xrange(size) if default_values[ii] != values2[ii]
    ]

    # The functions that changed when the variables in variable_indices2 were
    # changed cannot be functions of the variables in variable_indices1 (and
    # vice-versa).
    for jj in function_indices2:
      assignments[jj] -= set(variable_indices1)
    for jj in function_indices1:
      assignments[jj] -= set(variable_indices2)

    bit += bit

  for ii in xrange(size):
    if len(assignments[ii]) != 1:
      raise RuntimeError("unable to associate variables with functions")
    assignments[ii] = assignments[ii].pop()

  # We need to invert the permutation contained in "assignments", since we want
  # it to act on the functions, not the variables.
  permutation = [0] * size
  for ii in xrange(size):
    permutation[assignments[ii]] = ii

  return permutation


def find_zeros_of_functions(size, evaluate_fn, epsilon=1e-6):
  """Finds the zeros of a set of functions.

  Suppose that we have n functions and n variables, and that each of the former
  is a function of *exactly* one of the latter (we do *not* need to know which
  function depends on which variable: we'll figure it out). This situation
  commonly results from our use for thresholds and slack variables.

  This function will find a value for each variable that (rougly) causes the
  corresponding function to equal zero.

  Args:
    size: int, the number of functions and variables.
    evaluate_fn: function mapping a numpy array of variable values to another
      numpy array of function values. This must be a 1-1 mapping (i.e. this is
      just a bunch of 1d functions mashed together), but which input maps to
      which output need not be known (i.e. the inputs and outputs can be
      permuted arbitrarily).
    epsilon: float, the desired precision of the returned variable values.

  Returns:
    A numpy array of variable assignments that zero the functions.

  Raises:
    RuntimeError: if we fail to successfully find an assignment between
      variables and functions, or if the search fails to terminate.
  """
  # FUTURE WORK: make these into (optional) arguments.
  maximum_initialization_iterations = 16
  maximum_bisection_iterations = 64

  # Find the association between variables and functions.
  permutation = _associate_functions_with_variables(size, evaluate_fn)
  permuted_evaluate_fn = lambda values: evaluate_fn(values)[permutation]

  # Find initial lower/upper bounds for bisection search by growing the interval
  # until the two ends have different signs.
  lower_bounds = np.zeros(size)
  upper_bounds = np.ones(size)
  delta = 1.0
  iterations = 0
  while True:
    lower_values = permuted_evaluate_fn(lower_bounds)
    upper_values = permuted_evaluate_fn(upper_bounds)
    same_signs = ((lower_values > 0) == (upper_values > 0))
    if not any(same_signs):
      break
    lower_bounds[same_signs] -= delta
    upper_bounds[same_signs] += delta
    delta += delta

    iterations += 1
    if iterations > maximum_initialization_iterations:
      raise RuntimeError("initial lower/upper bound search did not terminate.")

  # Perform a bisection search to find the zeros.
  iterations = 0
  while True:
    middle_bounds = 0.5 * (lower_bounds + upper_bounds)
    middle_values = permuted_evaluate_fn(middle_bounds)
    same_signs = ((lower_values > 0) == (middle_values > 0))
    different_signs = ~same_signs
    lower_bounds[same_signs] = middle_bounds[same_signs]
    lower_values[same_signs] = middle_values[same_signs]
    upper_bounds[different_signs] = middle_bounds[different_signs]
    upper_values[different_signs] = middle_values[different_signs]
    if max(upper_bounds - lower_bounds) <= epsilon:
      break

    iterations += 1
    if iterations > maximum_bisection_iterations:
      raise RuntimeError("bisection search did not terminate")

  return 0.5 * (lower_bounds + upper_bounds)
