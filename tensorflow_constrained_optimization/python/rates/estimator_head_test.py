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
"""Tests for estimator_head.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import binary_rates
from tensorflow_constrained_optimization.python.rates import estimator_head
from tensorflow_constrained_optimization.python.rates import rate_minimization_problem
from tensorflow_constrained_optimization.python.rates import subsettable_context
from tensorflow_constrained_optimization.python.train import proxy_lagrangian_optimizer
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class HeadV2Test(tf.test.TestCase):
  """Tests for `HeadV2` in estimator_head.py."""

  def _get_inputs(self):
    """Generates constant inputs for tests."""
    # Data with one feature.
    features = np.reshape([-2.0, -0.4, 0.6], [-1, 1])

    # Labels are assigned so that a linear model that multiplies the feature
    # with a negative coefficient yields a recall of 100%.
    labels = np.reshape([1.0, 1.0, 0.0], [-1, 1])

    # Feature columns and dictionary needed for creating a tf.Estimator.
    feature_columns = [tf.feature_column.numeric_column("lone_feature")]
    features_dict = {"lone_feature": features}
    input_fn = lambda: (features_dict, labels)

    return input_fn, feature_columns

  def _recall_constrained_problem(self, recall_target):
    """Returns function returning a recall-constrained problem."""
    def problem_fn(logits, labels, features):
      # Returns a `RateMinimizationProblem` minimizing error rate subject to
      # recall >= recall_target.
      del features
      context = subsettable_context.rate_context(logits, labels)
      objective = binary_rates.error_rate(context)
      constraints = [binary_rates.true_positive_rate(context) >= recall_target]
      return rate_minimization_problem.RateMinimizationProblem(
          objective, constraints)
    return problem_fn

  def _train_and_evaluate_estimator(self, head, optimizer):
    """Trains `LinearEstimator` with head and optimizer, and returns metrics."""
    # Training data.
    input_fn, feature_columns = self._get_inputs()

    # Create Linear estimator.
    estimator = tf.estimator.LinearEstimator(
        head=head, feature_columns=feature_columns, optimizer=optimizer)

    # Add recall metric to estimator.
    def recall(labels, predictions, features):
      del features
      recall_metric = tf.keras.metrics.Recall(name="recall")
      probabilities = predictions["probabilities"][:, -1]
      recall_metric.update_state(y_true=labels, y_pred=probabilities)
      return {"recall": recall_metric}
    estimator = tf.estimator.add_metrics(estimator, recall)

    # Train estimator and evaluate it on the same data.
    estimator.train(input_fn=input_fn, max_steps=10)
    return estimator.evaluate(input_fn=input_fn, steps=1)

  def test_estimator_head_v2_with_constrained_optimizer(self):
    """Trains `Estimator` with `tfco.HeadV2` and a `ConstrainedOptimizerV2`."""
    # Create `tfco.HeadV2` instance with base binary head and constrained
    # optimization problem constraining recall to be at least 0.9.
    problem_fn = self._recall_constrained_problem(0.9)
    binary_head = tf.estimator.BinaryClassHead()
    head = estimator_head.HeadV2(binary_head, problem_fn)

    # Train and evaluate linear estimator with constrained optimizer, and assert
    # the recall for the trained model is at least 0.9.
    optimizer = proxy_lagrangian_optimizer.ProxyLagrangianOptimizerV2(
        optimizer=tf.keras.optimizers.Adagrad(1),
        constraint_optimizer=tf.keras.optimizers.Adagrad(1))
    results = self._train_and_evaluate_estimator(head, optimizer)
    self.assertGreaterEqual(results["recall"], 0.9)

  def test_estimator_head_v2_with_keras_optimizer(self):
    """Trains `Estimator` with `tfco.HeadV2` and a `tf.keras.Optimizer`."""
    # Create `tfco.HeadV2` with base binary head and constrained optimization
    # problem constraining recall to be at least 0.9.
    problem_fn = self._recall_constrained_problem(0.9)
    binary_head = tf.estimator.BinaryClassHead()
    head = estimator_head.HeadV2(binary_head, problem_fn)

    # Train and evaluate linear estimator with keras optimizer, and assert the
    # recall for the trained model is at least 0.9.
    optimizer = tf.keras.optimizers.Adagrad(1)
    results = self._train_and_evaluate_estimator(head, optimizer)
    self.assertGreaterEqual(results["recall"], 0.9)


if __name__ == "__main__":
  tf.test.main()
