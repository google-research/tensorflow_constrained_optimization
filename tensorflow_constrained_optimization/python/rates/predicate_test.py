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
"""Tests for predicate.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import predicate
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class PredicateTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for helper functions in predicate.py."""

  def test_predicate(self):
    """Tests the `Predicate` class."""
    memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.compat.v2.Variable(0, dtype=tf.int32)
    }

    predicate1 = predicate.Predicate([-0.2, 0.4, 1.0, 0.3])
    predicate2 = predicate.Predicate([0.8, 1.1, 0.6, 0.0])

    # We'll calculate the XOR of predicate1 and predicate2 in three ways. This
    # is the expected result.
    expected = [0.8, 0.6, 0.4, 0.3]

    actual1 = predicate1 ^ predicate2
    actual2 = (predicate1 & ~predicate2) | (~predicate1 & predicate2)
    actual3 = (predicate1 | predicate2) & ~(predicate1 & predicate2)

    with self.wrapped_session() as session:
      self.assertAllClose(
          expected, session.run(actual1.tensor(memoizer)), rtol=0, atol=1e-6)
      self.assertAllClose(
          expected, session.run(actual2.tensor(memoizer)), rtol=0, atol=1e-6)
      self.assertAllClose(
          expected, session.run(actual3.tensor(memoizer)), rtol=0, atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
