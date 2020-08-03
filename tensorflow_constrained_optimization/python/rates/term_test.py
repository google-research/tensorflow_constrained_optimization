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
"""Tests for term.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python import graph_and_eager_test_case
from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import deferred_tensor
from tensorflow_constrained_optimization.python.rates import loss
from tensorflow_constrained_optimization.python.rates import predicate
from tensorflow_constrained_optimization.python.rates import term
# Placeholder for internal import.


# @run_all_tests_in_graph_and_eager_modes
class TermTest(graph_and_eager_test_case.GraphAndEagerTestCase):
  """Tests for `_RatioWeights` and `Term` classes."""

  def __init__(self, *args, **kwargs):
    super(TermTest, self).__init__(*args, **kwargs)

    # We use a fixed fake dataset to make sure that the tests are reproducible.
    # The code for generating this random dataset is:
    #
    #   size = 20
    #   num_splits = 3
    #
    #   self._predictions = np.random.randn(size)
    #   self._weights = np.random.rand(size, 3)  # These weights must be >= 0.
    #   self._positive_weights = np.abs(np.random.randn(size, 3))
    #   self._negative_weights = np.abs(np.random.randn(size, 3))
    #
    #   numerator_count = int(size / 3)
    #   numerator_indices = set(random.sample(range(size), numerator_count))
    #   self._numerator_predicate = np.zeros(size, dtype=np.bool)
    #   for jj in numerator_indices:
    #     self._numerator_predicate[jj] = True
    #
    #   denominator_count = int(size / 3)
    #   denominator_indices = copy.copy(numerator_indices)
    #   denominator_indices.update(
    #       random.sample(range(size), denominator_count))
    #   denominator_indices.add(0)  # So the denominator cannot be zero.
    #   self._denominator_predicate = np.zeros(size, dtype=np.bool)
    #   for jj in denominator_indices:
    #     self._denominator_predicate[jj] = True
    #
    #   self._splits = sorted(random.sample(range(size - 1), num_splits - 1))
    #   self._splits = [0] + [split + 1 for split in self._splits] + [size]
    #
    # The dataset itself is:
    self._predictions = np.array([
        0.513897481971, 0.619094203661, -1.43614158309, 0.485346568967,
        0.501967872542, -1.87270885886, 0.329271973167, -1.39355007129,
        -0.809083865846, 0.0948141491137, 0.555680393836, -0.505485672514,
        1.2058450436, -1.22966914243, -0.712507031338, -0.254702125638,
        0.0898578087987, 0.721684390359, -0.258134491538, 0.930098073279
    ])
    self._weights = np.array([
        [0.0409076736058, 0.097025103707, 0.163097395161],
        [0.156324246476, 0.461219755708, 0.559603476089],
        [0.322678870155, 0.593875087267, 0.658791657311],
        [0.973160863197, 0.45641917991, 0.76456917351],
        [0.261461149623, 0.44133325217, 0.894653141497],
        [0.0455613908792, 0.00174937734628, 0.108379842179],
        [0.0817933091489, 0.459661219634, 0.0246866068574],
        [0.472988105844, 0.236402209366, 0.53693540622],
        [0.0113651938639, 0.0965270846054, 0.963704643542],
        [0.0569251847422, 0.79715387081, 0.211616608621],
        [0.277842672788, 0.731256815828, 0.154880690465],
        [0.144046346946, 0.688256584394, 0.929772651843],
        [0.702876957661, 0.263546618343, 0.161683272854],
        [0.961063032872, 0.0150274813334, 0.0402319212147],
        [0.56491082282, 0.528775415164, 0.628682464687],
        [0.628968240655, 0.00702453199629, 0.630627332423],
        [0.116267913574, 0.116785397481, 0.399121348291],
        [0.346990979889, 0.392097695551, 0.999308462825],
        [0.773288143924, 0.504101668364, 0.334371083275],
        [0.140311759457, 0.157850262752, 0.370491094206],
    ])
    self._positive_weights = np.array([
        [0.368879241431, 0.776283778201, 0.922791023591],
        [0.479167701857, 0.346558378353, 0.710694447574],
        [1.43551507315, 0.908609260426, 1.2048088488],
        [0.566753206511, 1.99042543647, 1.18133627163],
        [0.453050286383, 1.42491505353, 0.470967482097],
        [2.03454082335, 0.377569624532, 0.833301377769],
        [1.16157676163, 0.497598685605, 0.288470160892],
        [2.27255332987, 0.350871636523, 0.592990995235],
        [0.715961282376, 0.74996983227, 0.625752113108],
        [1.79041415884, 1.38971438305, 1.00465078918],
        [1.96777275689, 2.04654802919, 0.147776670171],
        [0.15385256971, 0.977058328072, 0.106955594983],
        [0.303448743856, 0.232478561899, 1.4492977573],
        [1.72497915945, 1.11767175531, 0.591949244909],
        [0.484786597662, 0.0477333948402, 1.02891263901],
        [1.11440680359, 0.0711480171322, 2.06228440953],
        [0.242549467822, 0.758649745143, 0.11736812012],
        [0.842942238643, 0.83167306903, 0.852899753813],
        [0.841449544315, 0.662098892203, 0.69720901344],
        [0.730600225006, 1.55963637073, 0.395150457709],
    ])
    self._negative_weights = np.array([
        [0.94119137969, 0.457577723139, 1.35379028598],
        [1.94403888363, 0.109948295025, 0.855261861779],
        [0.519643638412, 1.17592742071, 0.771248449529],
        [2.11864153683, 2.02543613619, 0.645954942694],
        [0.0482497767808, 1.28632126984, 0.00924264320825],
        [1.05315451703, 0.718584385367, 0.185098145457],
        [0.110022539127, 2.12282472908, 0.767980152035],
        [1.01391368028, 1.46060647161, 0.172538114564],
        [1.14967160138, 0.382091947887, 0.359917609479],
        [0.755304203266, 0.484833554786, 1.45780607174],
        [0.871276472033, 0.143186753181, 0.391562221716],
        [1.4504998766, 1.46002538742, 0.176054388245],
        [0.0475175253581, 0.0124053824983, 0.839016925593],
        [1.49997568438, 0.424100741707, 0.873922164513],
        [0.469077055039, 0.222382249017, 1.05446101882],
        [0.250843812691, 1.70255685948, 0.367081938199],
        [0.779788091612, 0.534183374808, 0.845266167632],
        [1.96668468564, 1.21420194107, 0.679016878896],
        [1.14222874548, 1.60752422957, 0.759153421009],
        [2.71462022507, 0.0845003688757, 2.11593571334],
    ])
    self._numerator_predicate = np.array([
        False, True, True, True, False, False, False, True, False, True, False,
        False, False, False, False, True, False, False, False, False
    ])
    self._denominator_predicate = np.array([
        True, True, True, True, True, False, False, True, False, True, False,
        False, False, True, False, True, False, False, True, False
    ])
    self._splits = [0, 9, 18, 20]

  def test_ratio_weights_zero(self):
    """Tests `_RatioWeights` with all-zero weights class method."""
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    ratio_weights = term._RatioWeights({}, 1)
    actual_weights = ratio_weights.evaluate(structure_memoizer)
    self.assertEqual(0, len(actual_weights.variables))

    # We don't need to run this in a session, since the expected weights are a
    # constant (zero).
    self.assertAllEqual([[0]], actual_weights(structure_memoizer))

  def test_ratio_weights_ratio(self):
    """Tests `_RatioWeights`'s ratio() class method."""
    weights_placeholder = self.wrapped_placeholder(tf.float32, shape=(None,))
    numerator_predicate_placeholder = self.wrapped_placeholder(
        tf.bool, shape=(None,))
    denominator_predicate_placeholder = self.wrapped_placeholder(
        tf.bool, shape=(None,))
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    numerator_predicate = predicate.Predicate(numerator_predicate_placeholder)
    denominator_predicate = predicate.Predicate(
        denominator_predicate_placeholder)
    ratio_weights = term._RatioWeights.ratio(
        deferred_tensor.ExplicitDeferredTensor(weights_placeholder), [1.0],
        numerator_predicate, denominator_predicate)
    actual_weights = ratio_weights.evaluate(structure_memoizer)

    # We need to explicitly create the variables before creating the wrapped
    # session.
    for variable in actual_weights.variables:
      variable.create(structure_memoizer)

    def update_ops_fn():
      update_ops = []
      for variable in actual_weights.variables:
        update_ops += variable.update_ops(structure_memoizer)
      return update_ops

    with self.wrapped_session() as session:
      running_count = 0.0
      running_sum = 0.0
      for ii in xrange(len(self._splits) - 1):
        begin_index = self._splits[ii]
        end_index = self._splits[ii + 1]
        size = end_index - begin_index

        weights_subarray = self._weights[begin_index:end_index, 0]
        numerator_predicate_subarray = self._numerator_predicate[
            begin_index:end_index]
        denominator_predicate_subarray = self._denominator_predicate[
            begin_index:end_index]

        running_count += size
        running_sum += np.sum(weights_subarray * denominator_predicate_subarray)
        average_denominator = running_sum / running_count
        expected_weights = (weights_subarray * numerator_predicate_subarray)
        expected_weights /= average_denominator

        # Running the update_ops will update the running denominator count/sum
        # calculated by the _RatioWeights object.
        session.run_ops(
            update_ops_fn,
            feed_dict={
                weights_placeholder:
                    weights_subarray,
                numerator_predicate_placeholder:
                    numerator_predicate_subarray,
                denominator_predicate_placeholder:
                    denominator_predicate_subarray
            })
        # Now we can calculate the weights.
        actual_weights_value = session.run(
            lambda: actual_weights(structure_memoizer),
            feed_dict={
                weights_placeholder:
                    weights_subarray,
                numerator_predicate_placeholder:
                    numerator_predicate_subarray,
                denominator_predicate_placeholder:
                    denominator_predicate_subarray
            })

        self.assertAllClose(
            np.expand_dims(expected_weights, axis=1),
            actual_weights_value,
            rtol=0,
            atol=1e-6)

        session.run_ops(
            lambda: structure_memoizer[defaults.GLOBAL_STEP_KEY].assign_add(1))

  def test_ratio_weights_arithmetic(self):
    """Tests `_RatioWeights`'s arithmetic operators."""
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    def create_ratio_weights(weights_tensor):
      return term._RatioWeights.ratio(
          deferred_tensor.ExplicitDeferredTensor(weights_tensor), [1.0],
          predicate.Predicate(True), predicate.Predicate(True))

    weights_tensors = [
        tf.constant(self._weights[:, ii], dtype=tf.float32) for ii in xrange(3)
    ]
    ratio_weights = [
        create_ratio_weights(weights_tensor)
        for weights_tensor in weights_tensors
    ]

    # We only check one particular expression, but all operations are exercised.
    ratio_weights = (-ratio_weights[0] + 0.3 * ratio_weights[1] -
                     ratio_weights[2] / 3.1 + ratio_weights[0] * 0.5)
    expected_weights = (-self._weights[:, 0] + 0.3 * self._weights[:, 1] -
                        self._weights[:, 2] / 3.1 + self._weights[:, 0] * 0.5)

    actual_weights = ratio_weights.evaluate(structure_memoizer)

    # We need to explicitly create the variables before creating the wrapped
    # session.
    for variable in actual_weights.variables:
      variable.create(structure_memoizer)

    with self.wrapped_session() as session:
      actual_weights_value = session.run(actual_weights(structure_memoizer))
      self.assertAllClose(
          np.expand_dims(expected_weights, axis=1),
          actual_weights_value,
          rtol=0,
          atol=1e-6)

  def test_ratio_weights_memoizer(self):
    """Tests memoization."""
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    weights_tensor = deferred_tensor.ExplicitDeferredTensor(
        tf.constant([0.5, 0.1, 1.0], dtype=tf.float32))
    numerator1_tensor = deferred_tensor.ExplicitDeferredTensor(
        tf.constant([True, False, True], dtype=tf.bool))
    numerator2_tensor = deferred_tensor.ExplicitDeferredTensor(
        tf.constant([True, True, False], dtype=tf.bool))
    numerator1_predicate = predicate.Predicate(numerator1_tensor)
    numerator2_predicate = predicate.Predicate(numerator2_tensor)
    denominator_predicate = predicate.Predicate(True)

    ratio_weights1 = term._RatioWeights.ratio(weights_tensor, [1.0],
                                              numerator1_predicate,
                                              denominator_predicate)
    ratio_weights2 = term._RatioWeights.ratio(weights_tensor, [1.0],
                                              numerator2_predicate,
                                              denominator_predicate)
    result1 = ratio_weights1.evaluate(structure_memoizer)
    result2 = ratio_weights2.evaluate(structure_memoizer)

    # The numerators differ, so the results should be different, but the
    # weights and denominators match, so the variables should be the same.
    self.assertIsNot(result1, result2)
    self.assertEqual(result1.variables, result2.variables)

  def test_binary_classification_term(self):
    """Tests `BinaryClassificationTerm`."""
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    def numpy_loss(positive_weights_array, negative_weights_array,
                   predictions_array):
      constant_weights = np.minimum(positive_weights_array,
                                    negative_weights_array)
      positive_weights = positive_weights_array - constant_weights
      negative_weights = negative_weights_array - constant_weights
      positive_losses = np.maximum(0.0, 1.0 + predictions_array)
      negative_losses = np.maximum(0.0, 1.0 - predictions_array)
      return np.mean(constant_weights + positive_weights * positive_losses +
                     negative_weights * negative_losses)

    def create_binary_classification_term(predictions_tensor,
                                          positive_weights_tensor,
                                          negative_weights_tensor):
      key = (1.0, predicate.Predicate(True))
      value = deferred_tensor.ExplicitDeferredTensor(
          tf.stack([positive_weights_tensor, negative_weights_tensor], axis=1))
      ratio_weights = term._RatioWeights({key: value}, 2)
      return term.BinaryClassificationTerm(
          deferred_tensor.ExplicitDeferredTensor(predictions_tensor),
          ratio_weights, loss.HingeLoss())

    # Use randomly constructed arrays of predictions and weights.
    predictions_tensor = tf.constant(self._predictions, dtype=tf.float32)
    positive_weights_tensors = [
        tf.constant(self._positive_weights[:, ii], dtype=tf.float32)
        for ii in xrange(3)
    ]
    negative_weights_tensors = [
        tf.constant(self._negative_weights[:, ii], dtype=tf.float32)
        for ii in xrange(3)
    ]

    term_objects = []
    for positive_weights_tensor, negative_weights_tensor in zip(
        positive_weights_tensors, negative_weights_tensors):
      term_object = create_binary_classification_term(predictions_tensor,
                                                      positive_weights_tensor,
                                                      negative_weights_tensor)
      term_objects.append(term_object)

    # We only check one particular expression, but all operations are exercised.
    term_object = (-term_objects[0] + 0.3 * term_objects[1] -
                   term_objects[2] / 3.1 + term_objects[0] * 0.5)
    expected_positive_weights = (-self._positive_weights[:, 0] +
                                 0.3 * self._positive_weights[:, 1] -
                                 self._positive_weights[:, 2] / 3.1 +
                                 self._positive_weights[:, 0] * 0.5)
    expected_negative_weights = (-self._negative_weights[:, 0] +
                                 0.3 * self._negative_weights[:, 1] -
                                 self._negative_weights[:, 2] / 3.1 +
                                 self._negative_weights[:, 0] * 0.5)
    expected_term = numpy_loss(expected_positive_weights,
                               expected_negative_weights, self._predictions)

    actual_term = term_object.evaluate(structure_memoizer)

    # We need to explicitly create the variables before creating the wrapped
    # session.
    for variable in actual_term.variables:
      variable.create(structure_memoizer)

    with self.wrapped_session() as session:
      actual_term_value = session.run(actual_term(structure_memoizer))
      self.assertAllClose(expected_term, actual_term_value, rtol=0, atol=1e-6)

  def test_multiclass_term(self):
    """Tests `MulticlassTerm`."""
    # This test is pretty much identical to test_binary_classification_term,
    # except that instead of using a BinaryClassificationTerm, we use a
    # MulticlassTerm with two classes.
    structure_memoizer = {
        defaults.DENOMINATOR_LOWER_BOUND_KEY: 0.0,
        defaults.GLOBAL_STEP_KEY: tf.Variable(0, dtype=tf.int32),
        defaults.VARIABLE_FN_KEY: tf.Variable
    }

    def numpy_loss(positive_weights_array, negative_weights_array,
                   predictions_array):
      constant_weights = np.minimum(positive_weights_array,
                                    negative_weights_array)
      positive_weights = positive_weights_array - constant_weights
      negative_weights = negative_weights_array - constant_weights
      positive_losses = np.maximum(0.0, 1.0 + predictions_array)
      negative_losses = np.maximum(0.0, 1.0 - predictions_array)
      return np.mean(constant_weights + positive_weights * positive_losses +
                     negative_weights * negative_losses)

    def create_multiclass_term(predictions_tensor, positive_weights_tensor,
                               negative_weights_tensor):
      key = (1.0, predicate.Predicate(True))
      value = deferred_tensor.ExplicitDeferredTensor(
          tf.stack([positive_weights_tensor, negative_weights_tensor], axis=1))
      ratio_weights = term._RatioWeights({key: value}, 2)
      return term.MulticlassTerm(
          deferred_tensor.ExplicitDeferredTensor(predictions_tensor),
          ratio_weights, loss.HingeLoss())

    # Use randomly constructed arrays of predictions and weights.
    predictions_tensor = tf.constant(
        np.stack([0.5 * self._predictions, -0.5 * self._predictions], axis=1),
        dtype=tf.float32)
    positive_weights_tensors = [
        tf.constant(self._positive_weights[:, ii], dtype=tf.float32)
        for ii in xrange(3)
    ]
    negative_weights_tensors = [
        tf.constant(self._negative_weights[:, ii], dtype=tf.float32)
        for ii in xrange(3)
    ]

    term_objects = []
    for positive_weights_tensor, negative_weights_tensor in zip(
        positive_weights_tensors, negative_weights_tensors):
      term_object = create_multiclass_term(predictions_tensor,
                                           positive_weights_tensor,
                                           negative_weights_tensor)
      term_objects.append(term_object)

    # We only check one particular expression, but all operations are exercised.
    term_object = (-term_objects[0] + 0.3 * term_objects[1] -
                   term_objects[2] / 3.1 + term_objects[0] * 0.5)
    expected_positive_weights = (-self._positive_weights[:, 0] +
                                 0.3 * self._positive_weights[:, 1] -
                                 self._positive_weights[:, 2] / 3.1 +
                                 self._positive_weights[:, 0] * 0.5)
    expected_negative_weights = (-self._negative_weights[:, 0] +
                                 0.3 * self._negative_weights[:, 1] -
                                 self._negative_weights[:, 2] / 3.1 +
                                 self._negative_weights[:, 0] * 0.5)
    expected_term = numpy_loss(expected_positive_weights,
                               expected_negative_weights, self._predictions)

    actual_term = term_object.evaluate(structure_memoizer)

    # We need to explicitly create the variables before creating the wrapped
    # session.
    for variable in actual_term.variables:
      variable.create(structure_memoizer)

    with self.wrapped_session() as session:
      actual_term_value = session.run(actual_term(structure_memoizer))
      self.assertAllClose(expected_term, actual_term_value, rtol=0, atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
