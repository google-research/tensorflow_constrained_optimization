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
"""TFCO estimator head for use with TensorFlow 1.x and 2.x.

The classes `HeadV1` and `HeadV2` allow you to create custom heads that can be
used with a `tf.Estimator` to minimize a constrained minimization problem
in TF 1.x and 2.x respectively. You would need to provide an existing `_Head`
for TF 1.x and a `tf.estimator.Head` for TF 2.x and a function specifying
a constrained minimization problem, and the base head's minimization ops
will be accordingly modified.

Example
=======
Consider a binary classification problem, where we wish to train a
LinearEstimator by minimizing error rate subject to a recall constraint. For
this, we will first create a function that takes "logits", "labels",
"features", and an (optional) "weight_column", and returns the
`RateMinimizationProblem`.
```python
   def problem_fn(logits, labels, features, weight_column=None):
     context = tfco.rate_context(predictions=logits, labels=labels)
     objective=tfco.error_rate(context)
     problem = tfco.RateMinimizationProblem(
       objective=tfco.error_rate(context),
       constraints=[tfco.recall(context) >= 0.9])
    return problem
```

In TF 2.x, we will then create a `tfco.HeadV2` instance from a base
`BinaryClassHead` instance and the `problem_fn` defined above.
```python
  base_head = tf.estimator.BinaryClassHead()
  custom_head = tfco.HeadV2(base_head, problem_fn)
```

The final step is to create a `LinearEstimator` using the custom head.
```python
  estimator = tf.estimator.LinearEstimator(head=custom_head, ...)
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_constrained_optimization.python.train import constrained_optimizer
from tensorflow_constrained_optimization.python.train import lagrangian_optimizer


class _WrapperHead(tf.estimator.Head):
  """Base class for `HeadV1` and `HeadV2`.

  This class is a wrapper around an existing base head, which can be either a
   V1 `_Head` instance or a V2 `Head` instance. While this class implements the
  `tf.estimator.Head` interface provided in TensorFlow 2.x, it can be used to
  wrap both a V1 and V2 head instance, as they have similar signatures. Some of
  the functions implemented may be relevant for only a V1 or V2 head.
  """

  def __init__(self, base_head, problem_fn, weight_column=None):
    """Initializes a `_WrapperHead` instance that wraps the `base_head`.

    Args:
      base_head: A V1 `_Head` instance or a V2 `tf.estimator.Head` instance.
      problem_fn: A function that takes logits, labels and features as input,
        and returns a `ConstrainedMinimizationProblem` instance.
      weight_column: Optional string or a `NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to re-weight examples during training. This may be
        an optional argument to the base_head, but however needs to be passed
        again here as `weight_column` is not usually publicly accessible in the
        base_head.
    """
    self._base_head = base_head
    self._problem_fn = problem_fn
    self._weight_column = weight_column

  @property
  def name(self):
    """The name of this head.

    Returns:
      A string.
    """
    return self._base_head.name

  @property
  def logits_dimension(self):
    """Size of the last dimension of the logits `Tensor`.

    Often is the number of classes, labels, or real values to be predicted.
    Typically, logits is of shape `[batch_size, logits_dimension]`.

    Returns:
      The expected size of the `logits` tensor.
    """
    return self._base_head.logits_dimension

  def create_loss(self, features, mode, logits, labels):
    """Returns a loss Tensor from provided logits.

    The returned loss is the same as that of the base head and is meant
    solely for book-keeping purposes. We do not return the custom loss used to
    create the train_op for constrained optimization, as this loss makes use of
    auxilliary variables whose values may not be set properly in EVAL mode. This
    function is relevant only for a V1 estimator.

    Args:
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` to be used for loss construction.
      labels: Labels `Tensor`, or `dict` of same.

    Returns:
      `LossSpec`.
    """
    return self._base_head.create_loss(labels, logits, features, mode)

  @property
  def loss_reduction(self):
    """One of `tf.losses.Reduction`.

    Returns the same value as the base head, and does not reflect how TFCO
    aggregates its losses internally. This function is relevant only for a V2
    estimator.

    Returns:
      The type of loss reduction used in the head.
    """
    # TODO: Should we return SUM_OVER_BATCH_SIZE, as this better
    #   represents TFCO's aggregation strategy, and may be used for rescaling
    #   purposes during distributed training?
    return self._base_head.loss_reduction

  def predictions(self, logits, keys=None):
    """Returns a `dict` of predictions from provided logits.

    This function is relevant only for a V2 estimator.

    Args:
      logits: Logits `Tensor` to be used for prediction construction.
      keys: A list of `string` for prediction keys. Defaults to `None`, meaning
        if not specified, predictions will be created for all the pre-defined
        valid keys in the head.

    Returns:
      A `dict` of predicted `Tensor` keyed by prediction name.
    """
    return self._base_head.predictions(logits, keys)

  def metrics(self, regularization_losses=None):
    """Returns a `dict` of metric objects.

    This function is relevant only for a V2 estimator.

    Args:
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
       A `dict` of metrics keyed by string name. The value is an instance of
       `Metric` class.
    """
    return self._base_head.metrics(regularization_losses)

  def update_metrics(self,
                     eval_metrics,
                     features,
                     logits,
                     labels,
                     mode=None,
                     regularization_losses=None):
    """Updates metric objects and returns a `dict` of the updated metrics.

    This function is relevant only for a V2 estimator.

    Args:
      eval_metrics: A `dict` of metrics to be updated.
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      logits: logits `Tensor` to be used for metrics update.
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      mode: Estimator's `ModeKeys`. In most cases, this arg is not used and can
        be removed in the method implementation.
      regularization_losses: A list of additional scalar losses to be added to
        the training and evaluation loss, such as regularization losses.  Note
        that, the `mode` arg is not used in the `tf.estimator.Head`. If the
        update of the metrics doesn't rely on `mode`, it can be safely ignored
        in the method signature.

    Returns:
       A `dict` of updated metrics keyed by name. The value is an instance of
       `Metric` class.
    """
    return self._base_head.update_metrics(
        eval_metrics, features, logits, labels, mode, regularization_losses)

  def loss(self,
           labels,
           logits,
           features=None,
           mode=None,
           regularization_losses=None):
    """Returns a loss `Tensor` from provided arguments.

    The returned loss is the same as that of the base head and is meant
    solely for book-keeping purposes. We do not return the custom loss used to
    create the train_op for constrained optimization, as this loss makes use of
    auxilliary variables whose values may not be set properly in EVAL mode. This
    function is relevant for a V2 estimator.

    Args:
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      logits: Logits `Tensor` to be used for loss construction.
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      mode: Estimator's `ModeKeys`. To be used in case loss calculation is
        different in Train and Eval mode.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      A scalar `Tensor` representing a dummy loss.
    """
    return self._base_head.loss(
        labels, logits, features, mode, regularization_losses)

  def _create_no_op_estimator_spec(self, features, mode, logits, labels):
    """Returns `EstimatorSpec` for the base head with no `train_op`."""
    return self._base_head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=logits,
        labels=labels,
        train_op_fn=lambda loss: tf.constant(0),
        regularization_losses=None)

  def _create_problem_and_update_optimizer(
      self, logits, labels, features, optimizer):
    """Returns `ConstrainedMinimizationProblem` created using `_problem_fn`."""
    problem = self._problem_fn(
        logits, labels, features, weight_column=self._weight_column)
    # Set the number of constraints in the optimizer. This is needed if
    # `num_constraints` hasn't already been specified to the optimizer, and
    # will also check that we aren't changing the number of constraints from
    # a previously-specified value.
    optimizer.num_constraints = problem.num_constraints
    return problem

  def _append_update_ops(self, train_op_fn, update_ops):
    """Returns `train_op` with control dependency on `update_ops`."""
    # We handle the case of update_ops=None separately because calling
    # tf.control_dependencies(None) in graph mode clears existing control
    # dependencies.
    if update_ops is None:
      train_op = train_op_fn()
    else:
      with tf.control_dependencies(update_ops):
        train_op = train_op_fn()
    return train_op


class HeadV1(_WrapperHead):
  """A wrapper around an existing V1 `_Head` for use with TensorFlow 1.x.

  This head modifies the base head's train_op to minimize a
  `ConstrainedMinimizationProblem` specified via `problem_fn`, while
  preserving the base head's other functionality. If the estimator's optimizer
  is an instance of `ConstrainedOptimizerV1`, it uses the same for minimization.
  If not, it creates a `ConstrainedOptimizerV1` instance from the estimator's
  optimizer.
  """

  def __init__(self, base_head, problem_fn, weight_column=None):
    """Initializes a `HeadV1` instance that wraps the `base_head`.

    Args:
      base_head: A `_Head` instance.
      problem_fn: A function that takes logits, labels and features as input,
        and returns a `ConstrainedMinimizationProblem` instance.
      weight_column: Optional string or a `NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to re-weight examples during training. This may be
        an optional argument to the base_head, but however needs to be passed
        again here as `weight_column` is not usually publicly accessible in the
        base_head.

    Raises:
      ValueError: If a `tf.estimator.Head` instance is passed as the
        `base_head`.
    """
    if isinstance(base_head, tf.estimator.Head):
      raise ValueError("You cannot pass a `tf.estimator.Head` instance as the "
                       "`base_head` to `HeadV1`.")
    super(HeadV1, self).__init__(base_head, problem_fn, weight_column)

  def create_estimator_spec(self,
                            features,
                            mode,
                            logits,
                            labels=None,
                            optimizer=None,
                            train_op_fn=None,
                            regularization_losses=None):
    """Returns `EstimatorSpec` that a model_fn can return.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` to be used by the head.
      labels: Optional labels `Tensor`, or `dict` mapping string label names to
        `Tensor` objects of the label values.
      optimizer: A `ConstrainedOptimizerV1` or a `tf.compat.v1.train.Optimizer`.
        instance. If a `tf.compat.v1.train.Optimizer` is provided, the head
        creates a `ConstrainedOptimizerV1` that wraps it. This is an optional
        argument in the `_Head` base class, but needs to be passed here.
      train_op_fn: This argument is ignored and can be left unspecified.
      regularization_losses: This argument is ignored and can be left
        unspecified.

    Returns:
      `EstimatorSpec`.
    """
    estimator_spec = self._create_no_op_estimator_spec(
        features=features,
        mode=mode,
        logits=logits,
        labels=labels)

    # When mode is PREDICT or EVAL, no modification needed to the base head.
    if (mode == tf.estimator.ModeKeys.PREDICT) or (
        mode == tf.estimator.ModeKeys.EVAL):
      return estimator_spec

    # When mode is TRAIN, replace train_op in estimator_spec.
    if mode == tf.estimator.ModeKeys.TRAIN:
      if optimizer is None:
        raise ValueError("You must provide an optimizer to the estimator.")
      # TODO: Add support for passing train_op_fn.

      # If the optimizer is not a `ConstrainedOptimizerV1` instance, then
      # create a `LagrangianOptimizerV1` that wraps the base heads's optimizer.
      if not isinstance(
          optimizer, constrained_optimizer.ConstrainedOptimizerV1):
        optimizer = lagrangian_optimizer.LagrangianOptimizerV1(optimizer)

      problem = self._create_problem_and_update_optimizer(
          logits, labels, features, optimizer)

      # Create `train_op` with a control dependency on `UPDATE_OPS`.
      update_ops = tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.UPDATE_OPS)
      global_step = tf.compat.v1.train.get_global_step()
      train_op = self._append_update_ops(
          lambda: optimizer.minimize(problem, global_step=global_step),
          update_ops)
      return estimator_spec._replace(train_op=train_op)

    raise ValueError("mode={} not recognized".format(mode))


class HeadV2(_WrapperHead):
  """A wrapper around an existing V2 `Head` for use with TensorFlow 2.x.

  This head modifies the base head's train_op to minimize a
  `ConstrainedMinimizationProblem` specified via `problem_fn`, while
  preserving the base head's other functionality. If the estimator's optimizer
  is an instance of `ConstrainedOptimizerV2`, it uses the same for minimization.
  If not, it creates a `ConstrainedOptimizerV2` instance from the estimator's
  optimizer.
  """

  def __init__(self, base_head, problem_fn, weight_column=None):
    """Initializes a `HeadV2` instance that wraps the `base_head`.

    Args:
      base_head: A `tf.estimator.Head` instance.
      problem_fn: A function that takes logits, labels and features as input,
        and returns a `ConstrainedMinimizationProblem` instance.
      weight_column: Optional string or a `NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to re-weight examples during training. This may be
        an optional argument to the base_head, but however needs to be passed
        again here as `weight_column` is not usually publicly accessible in the
        base_head.

    Raises:
      ValueError: If a `tf.estimator.Head` instance is not passed as
        the `base_head`.
    """
    if not isinstance(base_head, tf.estimator.Head):
      raise ValueError("You must pass a `tf.estimator.Head` instance as "
                       "`base_head` to `HeadV2`.")
    super(HeadV2, self).__init__(base_head, problem_fn, weight_column)

  def create_estimator_spec(self,
                            features,
                            mode,
                            logits,
                            labels=None,
                            optimizer=None,
                            trainable_variables=None,
                            train_op_fn=None,
                            update_ops=None,
                            regularization_losses=None):
    """Returns `EstimatorSpec` for constrained optimization.

    The `EstimatorSpec` is the same as that of the base head with the
    `train_op` alone replaced with one for minimizing the constrained
    minimization problem specified by self._problem_fn.

    Args:
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      mode: Estimator's `ModeKeys`.
      logits: Logits `Tensor` to be used by the head.
      labels: Optional labels `Tensor`, or `dict` mapping string label names to
        `Tensor` objects of the label values.
      optimizer: A `ConstrainedOptimizerV2` or a `tf.keras.optimizers.Optimizer`
        instance. If a `tf.keras.optimizers.Optimizer` is provided, the head
        creates a `ConstrainedOptimizerV2` that wraps it. This is an optional
        argument in the `tf.estimator.Head` base class, but needs to be passed
        here.
      trainable_variables: A list or tuple of `Variable` objects to update to
        solve the constrained minimization problem. In Tensorflow 1.x, by
        default these are the list of variables collected in the graph under the
        key `GraphKeys.TRAINABLE_VARIABLES`. As Tensorflow 2.x doesn't have
        collections and GraphKeys, trainable_variables needs to be passed
        explicitly here.
      train_op_fn: This argument is ignored and can be left unspecified.
      update_ops: Optional list or tuple of update ops to be run at training
        time. For example, layers such as BatchNormalization create mean and
        variance update ops that need to be run at training time. In Tensorflow
        1.x, these are thrown into an UPDATE_OPS collection. As Tensorflow 2.x
        doesn't have collections, update_ops needs to be passed explicitly here.
      regularization_losses: This argument is ignored and can be left
        unspecified.

    Returns:
      `EstimatorSpec`.

    Raises:
      ValueError: If mode is not recognized or optimizer is not specified in
        TRAIN mode.
    """
    estimator_spec = self._create_no_op_estimator_spec(
        features=features,
        mode=mode,
        logits=logits,
        labels=labels)

    # When mode is PREDICT or EVAL, no modification needed to the base head.
    if (mode == tf.estimator.ModeKeys.PREDICT) or (
        mode == tf.estimator.ModeKeys.EVAL):
      return estimator_spec

    # When mode is TRAIN, replace train_op in estimator_spec.
    if mode == tf.estimator.ModeKeys.TRAIN:
      if optimizer is None:
        raise ValueError("You must provide an optimizer to the estimator.")
      # TODO: Add support for passing train_op_fn.

      # If the optimizer is not a `ConstrainedOptimizerV2` instance, then
      # create a `LagrangianOptimizer` that wraps the base heads's optimizer.
      if not isinstance(
          optimizer, constrained_optimizer.ConstrainedOptimizerV2):
        iterations = optimizer.iterations
        optimizer = lagrangian_optimizer.LagrangianOptimizerV2(optimizer)
        # Pass the iterations member (which contains the global step) in the
        # base head's optimizer to the newly created one.
        optimizer.iterations = iterations

      problem = self._create_problem_and_update_optimizer(
          logits, labels, features, optimizer)
      # Create `train_op` with a control dependency on the `update_ops`.
      var_list = trainable_variables + list(
          problem.trainable_variables) + optimizer.trainable_variables()
      train_op = self._append_update_ops(
          lambda: tf.group(optimizer.get_updates(problem, var_list)),
          update_ops)
      return estimator_spec._replace(train_op=train_op)

    raise ValueError("mode={} not recognized".format(mode))
