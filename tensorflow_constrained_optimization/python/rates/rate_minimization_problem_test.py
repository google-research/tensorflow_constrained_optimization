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
"""Tests for rate_minimization_problem.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import binary_rates
from tensorflow_constrained_optimization.python.rates import expression
from tensorflow_constrained_optimization.python.rates import rate_minimization_problem
from tensorflow_constrained_optimization.python.rates import subsettable_context
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class RateMinimizationProblemTest(
    graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for `RateMinimizationProblem` class."""

  def __init__(self, *args, **kwargs):
    super(RateMinimizationProblemTest, self).__init__(*args, **kwargs)

    # We use a list of fixed fake datasets to make sure that the tests are
    # reproducible.
    self._predictions = [
        np.array([0.34609016, 2.67008472, -0.50792732, -1.14588047,
                  0.02408896]),
        np.array([
            0.13728618, -1.08673067, 1.654064, -0.16789998, -1.14599855,
            -0.75179404, -0.055209
        ]),
        np.array([
            0.22335254, 0.25865217, -0.81673995, 0.09589956, -0.80554226,
            0.75360623, -0.62946276, 0.239591, 1.11925537, -1.43435933,
            0.28400089
        ]),
        np.array([
            -0.03203334, 2.01618121, -1.80739938, 1.53923583, 1.66509667,
            -0.54951989, 0.72464518, 0.23044844, -1.14854705, -0.40375554,
            0.23631839, 1.90914763, -1.31180266
        ])
    ]
    self._weights = [
        np.array([0.67997776, 0.17073744, 0.94912975, 0.05620256, 0.82680132]),
        np.array([
            0.31465006, 0.55479958, 0.90828850, 0.89661150, 0.32880753,
            0.33482013, 0.00679486
        ]),
        np.array([
            0.27611168, 0.49502870, 0.63035543, 0.84390883, 0.68964628,
            0.46231374, 0.17826599, 0.15767450, 0.35104775, 0.18909999,
            0.97175554
        ]),
        np.array([
            0.14981442, 0.80307177, 0.32802114, 0.05721732, 0.86284292,
            0.42942430, 0.80011760, 0.36128284, 0.72950361, 0.07733621,
            0.50577070, 0.39348359, 0.67013457
        ])
    ]
    self._num_datasets = 4

  @property
  def _contexts(self):
    # We can't create the contexts in __init__, since they would then wind up in
    # the wrong TensorFlow graph.
    result = []
    for index in xrange(self._num_datasets):

      # To support eager mode, the predictions should be a nullary function
      # returning a Tensor, but we need to make sure that it has its own copy of
      # "index".
      def predictions(index=index):
        return tf.constant(self._predictions[index], dtype=tf.float32)

      # In support eager mode, the weights should be a nullary function
      # returning a Tensor, but we need to make sure that it has its own copy of
      # "index".
      def weights(index=index):
        return tf.constant(self._weights[index], dtype=tf.float32)

      result.append(
          subsettable_context.rate_context(
              predictions=predictions, weights=weights))

    return result

  def test_minimization_problem_construction(self):
    """Checks that `RateMinimizationProblem`s are constructed correctly."""
    denominator_lower_bound = 0.0

    penalty_positive_prediction_rates = []
    constraint_positive_prediction_rates = []
    for index in xrange(self._num_datasets):
      predictions = self._predictions[index]
      weights = self._weights[index]
      # For the penalties, the default loss is hinge. These will be used for the
      # "objective" and "proxy_constraints" in the resulting
      # RateMinimizationProblem.
      penalty_positive_prediction_rates.append(
          np.sum(weights * np.maximum(0.0, 1.0 + predictions)) /
          np.sum(weights))
      # For the constraints, the default loss is zero-one. These will be used
      # for the "constraints" in the resulting RateMinimizationProblem.
      constraint_positive_prediction_rates.append(
          np.sum(weights * 0.5 * (1.0 + np.sign(predictions))) /
          np.sum(weights))

    contexts = self._contexts

    # Construct the objective and three constraints. These are as simple as
    # possible, since rates and constraints are tested elsewhere.
    objective = binary_rates.positive_prediction_rate(contexts[0])
    constraint1 = binary_rates.positive_prediction_rate(contexts[1])
    constraint2 = binary_rates.positive_prediction_rate(contexts[2])
    constraint3 = binary_rates.positive_prediction_rate(contexts[3])

    # Make the objective and constraints include each other as:
    #   objective contains constraint2, which contains constraint3
    #   constraint1 contains constraint3
    #   constraint2 contains constraint3
    # Notice that constraint3 is contained in both constraint1 and constraint2,
    # and indirectly in the objective. Despite this, we only want it to occur
    # once in the resulting optimization problem.
    constraint3 = constraint3 <= 0
    constraint1 = (
        expression.ConstrainedExpression(
            constraint1, extra_constraints=[constraint3]) <= 0)
    constraint2 = (
        expression.ConstrainedExpression(
            constraint2, extra_constraints=[constraint3]) <= 0)
    objective = expression.ConstrainedExpression(
        objective, extra_constraints=[constraint2])

    problem = rate_minimization_problem.RateMinimizationProblem(
        objective, [constraint1],
        denominator_lower_bound=denominator_lower_bound)

    with self.wrapped_session() as session:
      # In eager mode, tf.Variable.initial_value doesn't work, so we need to
      # cache the initial values for later.
      if tf.executing_eagerly():
        variables_and_initial_values = [
            (variable, variable.numpy()) for variable in problem.variables
        ]

      initial_objective = session.run(problem.objective())
      initial_constraints = session.run(problem.constraints())
      initial_proxy_constraints = session.run(problem.proxy_constraints())

      # We only need to run the update ops once, since the entire dataset is
      # contained within the Tensors, so the denominators will be correct.
      session.run_ops(problem.update_ops)

      actual_objective = session.run(problem.objective())
      actual_constraints = session.run(problem.constraints())
      actual_proxy_constraints = session.run(problem.proxy_constraints())

      # If we update the internal state, and then re-initialize, then the
      # resulting objective, constraints and proxy constraints should be the
      # same as they were before running the update_ops.
      if tf.executing_eagerly():
        for variable, initial_value in variables_and_initial_values:
          variable.assign(initial_value)
      else:
        session.run_ops(
            lambda: tf.compat.v1.variables_initializer(problem.variables))

      reinitialized_objective = session.run(problem.objective())
      reinitialized_constraints = session.run(problem.constraints())
      reinitialized_proxy_constraints = session.run(problem.proxy_constraints())

    self.assertEqual(3, initial_constraints.size)
    self.assertEqual(3, initial_proxy_constraints.size)
    self.assertEqual(3, actual_constraints.size)
    self.assertEqual(3, actual_proxy_constraints.size)
    self.assertEqual(3, reinitialized_constraints.size)
    self.assertEqual(3, reinitialized_proxy_constraints.size)

    # The first context is used for the objective.
    self.assertAllClose(
        penalty_positive_prediction_rates[0],
        actual_objective,
        rtol=0,
        atol=1e-6)
    # The last three contexts are used for the constraints. They'll occur in no
    # particular order, so we sort before comparing.
    self.assertAllClose(
        sorted(constraint_positive_prediction_rates[1:]),
        sorted(actual_constraints),
        rtol=0,
        atol=1e-6)
    self.assertAllClose(
        sorted(penalty_positive_prediction_rates[1:]),
        sorted(actual_proxy_constraints),
        rtol=0,
        atol=1e-6)

    # Make sure that, after re-initialization, we get the same results as we did
    # before executing the update_ops.
    self.assertAllClose(
        initial_objective, reinitialized_objective, rtol=0, atol=1e-6)
    self.assertAllClose(
        initial_constraints, reinitialized_constraints, rtol=0, atol=1e-6)
    self.assertAllClose(
        initial_proxy_constraints,
        reinitialized_proxy_constraints,
        rtol=0,
        atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
