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
"""Tests for candidates.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python import candidates


class CandidatesTest(tf.test.TestCase):

  # Set up the problem that will be used for most of the tests.
  _objective_vector = np.array(
      [0.03053309, -0.06667082, 0.88355145, 0.46529806])
  _constraints_matrix = np.array([[-0.60164551, 0.00371592],
                                  [0.36676229, -0.16392108],
                                  [0.7856454, -0.59778071],
                                  [-0.8441711, -0.56908492]])

  def test_inconsistent_shapes_for_best_distribution(self):
    """An error is raised when parameters have inconsistent shapes."""
    objective_vector = np.array([1, 2, 3])
    constraints_matrix = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])
    with self.assertRaises(ValueError):
      _ = candidates.find_best_candidate_distribution(objective_vector,
                                                      constraints_matrix)

  def test_inconsistent_shapes_for_best_index(self):
    """An error is raised when parameters have inconsistent shapes."""
    objective_vector = np.array([1, 2, 3])
    constraints_matrix = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])
    with self.assertRaises(ValueError):
      _ = candidates.find_best_candidate_index(objective_vector,
                                               constraints_matrix)

  def test_best_distribution(self):
    """Distribution should match known solution."""
    distribution = candidates.find_best_candidate_distribution(
        self._objective_vector, self._constraints_matrix)
    # Verify that the solution is a probability distribution.
    self.assertTrue(np.all(distribution >= -1e-6))
    self.assertAlmostEqual(np.sum(distribution), 1.0)
    # Verify that the solution satisfies the constraints.
    maximum_constraint_violation = np.amax(
        np.tensordot(self._constraints_matrix, distribution, axes=(0, 0)))
    self.assertLessEqual(maximum_constraint_violation, 1e-6)
    # Verify that the solution matches that which we expect.
    expected_distribution = np.array([0.37872711, 0.62127289, 0, 0])
    self.assertAllClose(expected_distribution, distribution, rtol=0, atol=1e-6)

  def test_best_index_rank_objectives_true_max_constraints_false(self):
    """Index should match known solution."""
    # To see why the expected answer is 1, notice that:
    #   self._objective_vector ranks = [2, 1, 4, 3].
    #   self._constraints_matrix ranks = [[1, 4], [3, 1], [4, 1], [1, 1]].
    #   Maximum ranks = [4, 3, 4, 3].
    index = candidates.find_best_candidate_index(
        self._objective_vector,
        self._constraints_matrix,
        rank_objectives=True,
        max_constraints=False)
    self.assertEqual(1, index)

  def test_best_index_rank_objectives_true_max_constraints_true(self):
    """Index should match known solution."""
    # To see why the expected answer is 0, notice that:
    #   self._objective_vector ranks = [2, 1, 4, 3].
    #   self._constraints_matrix maximum ranks = [2, 3, 4, 1]
    index = candidates.find_best_candidate_index(
        self._objective_vector,
        self._constraints_matrix,
        rank_objectives=True,
        max_constraints=True)
    self.assertEqual(0, index)

  def test_best_index_rank_objectives_false_max_constraints_false(self):
    """Index should match known solution."""
    # To see why the expected answer is 3, notice that:
    #   self._objective_vector ranks = [2, 1, 4, 3].
    #   self._constraints_matrix ranks = [[1, 4], [3, 1], [4, 1], [1, 1]].
    #   Maximum ranks = [4, 3, 4, 1].
    index = candidates.find_best_candidate_index(
        self._objective_vector,
        self._constraints_matrix,
        rank_objectives=False,
        max_constraints=False)
    self.assertEqual(3, index)

  def test_best_index_rank_objectives_false_max_constraints_true(self):
    """Index should match known solution."""
    # To see why the expected answer is 3, notice that:
    #   self._objective_vector ranks = [2, 1, 4, 3].
    #   self._constraints_matrix maximum ranks = [2, 3, 4, 1]
    index = candidates.find_best_candidate_index(
        self._objective_vector,
        self._constraints_matrix,
        rank_objectives=False,
        max_constraints=True)
    self.assertEqual(3, index)


if __name__ == "__main__":
  tf.test.main()
