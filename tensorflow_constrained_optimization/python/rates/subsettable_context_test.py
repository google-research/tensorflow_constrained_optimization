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
"""Tests for subsettable_context.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import subsettable_context


def create_contexts():
  """Returns a pair of `SubsettableContext`s to use in tests.

  We'll refer to the two contexts as "context1" and "context2". Both are subsets
  of the same parent context, which has:
    penalty_predicate    = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    constraint_predicate = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  context1 is subsetted from the parent context using:
    penalty_predicate1    = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    constraint_predicate1 = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
  while context2 is subsetted deom the parent context using:
    penalty_predicate2    = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    constraint_predicate2 = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]

  Returns:
    The pair (context1, context2).
  """
  predictions = tf.constant(0.0, dtype=tf.float32, shape=(1,))
  context = subsettable_context.rate_context(predictions)

  penalty_predicate = tf.constant(
      [True, False, True, False, True, False, True, False, True, False],
      dtype=tf.bool)
  constraint_predicate = tf.constant(
      [False, True, False, True, False, True, False, True, False, True],
      dtype=tf.bool)
  context = context.subset(penalty_predicate, constraint_predicate)

  penalty_predicate1 = tf.constant(
      [False, False, True, True, True, True, False, False, False, False],
      dtype=tf.bool)
  constraint_predicate1 = tf.constant(
      [True, True, False, False, False, False, True, True, True, True],
      dtype=tf.bool)

  penalty_predicate2 = tf.constant(
      [False, False, False, False, True, True, True, True, False, False],
      dtype=tf.bool)
  constraint_predicate2 = tf.constant(
      [True, True, True, True, False, False, False, False, True, True],
      dtype=tf.bool)

  context1 = context.subset(penalty_predicate1, constraint_predicate1)
  context2 = context.subset(penalty_predicate2, constraint_predicate2)

  return context1, context2


class SubsettableContextTest(tf.test.TestCase):
  """Tests for `SubsettableContext` class."""

  def test_subset_of_subset(self):
    """Tests that taking the subset-of-a-subset works correctly."""
    context1, context2 = create_contexts()

    context1_penalty_predicate = context1.penalty_predicate.predicate
    context1_constraint_predicate = context1.constraint_predicate.predicate
    context2_penalty_predicate = context2.penalty_predicate.predicate
    context2_constraint_predicate = context2.constraint_predicate.predicate

    with self.session() as session:
      session.run(tf.global_variables_initializer())

      # Make sure that the subset of a subset ANDs the conditions together in
      # condition1.
      expected_penalty_predicate = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                                            dtype=np.float32)
      expected_constraint_predicate = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                                               dtype=np.float32)
      self.assertAllEqual(expected_penalty_predicate,
                          session.run(context1_penalty_predicate))
      self.assertAllEqual(expected_constraint_predicate,
                          session.run(context1_constraint_predicate))
      # Likewise in condition2.
      expected_penalty_predicate = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                                            dtype=np.float32)
      expected_constraint_predicate = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                                               dtype=np.float32)
      self.assertAllEqual(expected_penalty_predicate,
                          session.run(context2_penalty_predicate))
      self.assertAllEqual(expected_constraint_predicate,
                          session.run(context2_constraint_predicate))

  def test_and(self):
    """Tests `SubsettableContext`'s logical AND operator."""
    context1, context2 = create_contexts()

    and_context = context1 & context2

    and_context_penalty_predicate = and_context.penalty_predicate.predicate
    and_context_constraint_predicate = (
        and_context.constraint_predicate.predicate)

    with self.session() as session:
      session.run(tf.global_variables_initializer())

      # Make sure that AND applies only to the top-level subset.
      expected_penalty_predicate = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                            dtype=np.float32)
      expected_constraint_predicate = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                               dtype=np.float32)
      self.assertAllEqual(expected_penalty_predicate,
                          session.run(and_context_penalty_predicate))
      self.assertAllEqual(expected_constraint_predicate,
                          session.run(and_context_constraint_predicate))

  def test_or(self):
    """Tests `SubsettableContext`'s logical OR operator."""
    context1, context2 = create_contexts()

    or_context = context1 | context2

    or_context_penalty_predicate = or_context.penalty_predicate.predicate
    or_context_constraint_predicate = or_context.constraint_predicate.predicate

    with self.session() as session:
      session.run(tf.global_variables_initializer())

      # Make sure that OR applies only to the top-level subset.
      expected_penalty_predicate = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                                            dtype=np.float32)
      expected_constraint_predicate = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                               dtype=np.float32)
      self.assertAllEqual(expected_penalty_predicate,
                          session.run(or_context_penalty_predicate))
      self.assertAllEqual(expected_constraint_predicate,
                          session.run(or_context_constraint_predicate))


if __name__ == "__main__":
  tf.test.main()
