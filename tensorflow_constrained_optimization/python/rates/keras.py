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
"""Contains helpers for integrating with Keras.

The classes in this file are intended to make it easier to work with TFCO and
Keras. In particular, they make it possible to use them together with Keras'
standard `Model.compile` and `Model.fit` methods, instead of a custom training
loop.

A `KerasPlaceholder` defines an input to TFCO, from Keras. It is initialized
from a function that takes two parameters---y_true and y_pred, as in a Keras
loss function---and returns the value that should be contained in this
placeholder. These objects, once constructed, are used to define the inputs to
TFCO contexts.

The other main class, `KerasLayer`, is a "dummy" layer that should be inserted
into the Keras model sequence. From the perspective of the model, it's an
identity function, but it contains all of the internal state needed by TFCO
(most notably, the Lagrange multipliers).

Example
=======

Imagine that we wish, for a standard binary classification problem, to minimize
the error rate subject to a recall constraint. To begin with, we will define a
context for which its "predictions" are taken from the "y_pred" loss function
parameter, and its "labels" from the "y_true" parameter:

```python
predictions = tfco.KerasPlaceholder(lambda _, y_pred: y_pred)
labels = tfco.KerasPlaceholder(lambda y_true, _: y_true)

context = tfco.rate_context(predictions=predictions, labels=labels)
```

If you want to use additional side features (e.g. per-example weights, protected
class memberships, the predictions of a baseline model, etc.), then you can
place them in the labels passed to Keras, and extract them from the "y_true"
parameter passed to the function from which your `KerasPlaceholder` is
constructed.

We next construct the `KerasLayer`, which plays a similar role, here, to that
played by a `RateMinimizationProblem` in "raw" TensorFlow optimization:

```python
layer = tfco.KerasLayer(
    objective=tfco.error_rate(context),
    constraints=[tfco.recall(context) >= 0.9],
    placeholders=[predictions, labels])
````

The `KerasLayer` must be included in the sequential model that we then create.
This is an example, so we use a simple linear model (here, "dimension" is the
dimension of the input feature vectors). The "layer" can be placed anywhere in
the sequence, since from the perspective of the model, it's just an identity
function. It needs to be *present*, since it contains the additional
dependencies required by a TFCO loss function, but it doesn't actually *do*
anything to its inputs:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=None, input_shape=(dimension,)),
    layer
])
```

The next step is to compile the model. This works just as it would for a normal
Keras model, except that the loss function is extracted from "layer", and will
correctly update not only the model parameters themselves, but also the internal
TFCO state (Lagrange multipliers, slack variables, etc.) during training, in
such a way as to optimize the constrained problem that was passed to the
`KerasLayer`'s constructor.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=1.0),
    loss=tfco_layer.loss)
```

At this point, you can call `Model.fit` just as you normally would.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python.rates import rate_minimization_problem
from tensorflow_constrained_optimization.python.train import lagrangian_optimizer


# This object doesn't inherit from helpers.RateObject since it isn't exactly an
# "internal" object: these are created by *users*, and passed to contexts.
class KerasPlaceholder(object):
  """Represents a Keras input to a TFCO constrained optimization problem.

  TFCO basically implements a special loss layer for constrained optimization.
  One issue with this, when it comes to interfacing with Keras, is that we need
  a place to "put" the Lagrange multipliers, slack variables, and so on---this
  role is played by `KerasLayer`. The `KerasPlaceholder` class deals with the
  other main issue: inputs.

  The basic problem is that, in TFCO, we set up the constrained optimization
  problem (and therefore the loss function) *outside* of the main training loop.
  We need a way to get the predictions and labels, which are passed to the Keras
  loss, to the TFCO problem. To do this, we create these placeholder classes,
  which are parameterized by a *function* that takes the labels "y_true" and
  predictions "y_pred" as inputs, and returns whatever slice of these inputs
  should be represented by the placeholder.

  In the simplest cases, you'll create two such placeholders, one of which
  returns "y_pred" and ignores "y_true", and the other of which does the
  opposite. You will then create a context with these as its "predictions" and
  "labels" parameters, respectively.

  In more complex cases, e.g. in a fairness problem, in which protected group
  membership information must be available, you can place these features in
  extra label columns (i.e. extra columns of "y_true"), and then extract the
  relevant features from "y_true" in the placeholder.
  """

  def __init__(self, callback):
    """Creates a new `KerasPlaceholder`.

    Args:
      callback: a function that takes two `Tensor` parameters, "y_true" and
        "y_pred", containing the labels and predictions of the model,
        respectively, and returns the `Tensor` that should be represented by
        this placeholder.
    """
    self._callback = callback
    self.assign()

  def __call__(self):
    if self._y_true is None or self._y_pred is None:
      raise RuntimeError("tfco.KerasPlaceholder was not assigned---please "
                         "check that the \"placeholders\" parameter to your "
                         "KerasLayer includes all input placeholders")
    return self._callback(self._y_true, self._y_pred)

  def assign(self, y_true=None, y_pred=None):
    self._y_true = y_true
    self._y_pred = y_pred


class _KerasInitializer(tf.keras.initializers.Initializer):
  """Internal Keras `Initializer` for KerasLayer.

  This `Initializer` plays essentially the same role as
  `tf.keras.initializers.Constant', except that whereas 'Constant' only accepts
  scalars, this `Initializer` works for higher-rank numpy arrays.
  """

  def __init__(self, value):
    """Creates a new `_KerasInitializer`.

    Args:
      value: scalar or numpy array, the initial value of the `Tensor` being
        initialized.
    """
    super(_KerasInitializer, self).__init__()
    self._value = value

  def __call__(self, shape, dtype=None):
    return tf.constant(value=self._value, dtype=dtype, shape=shape)

  def get_config(self):
    return {"value": self._value}

  # The default implementation of from_config is fine.


class KerasLayer(tf.keras.layers.Layer):
  """Dummy `Layer` containing internal TFCO state.

  TFCO basically implements a special loss layer for constrained optimization.
  One issue with this, when it comes to interfacing with Keras, is that we need
  a place to "put" the Lagrange multipliers, slack variables, and so on---this
  role is played by `KerasLayer`, which can be thought of as essentially a
  wrapper around a `RateMinimizationProblem` (although it does contain some
  extra information: see the "loss_creator" constructor parameter).

  A TFCO `RateMinimizationProblem` contains a number of hidden auxiliary
  variables, among them slack variables, and rate denominators. Likewise, a
  TFCO loss (or `Optimizer`) contains Lagrange multipliers, or another set of
  variables that plays the same role (as in the proxy-Lagrangian formulation).

  In Keras, we need to store these variables somewhere, so this dummy `Layer`
  creates and owns them. From the perspective of the model, it's a NO-OP (more
  accurately, an identity function), but it must be included in order for the
  internal TFCO variables to be updated during training.

  The "loss" method is the loss that should be passed to `Model.compile`:
  it is this loss function that ties together all of these auxiliary quantities,
  and causes the desired constrained problem to be solved.

  You should be aware that the "get_config" and "from_config" members will strip
  out all of this internal state. After saving+loading a model including a
  `KerasLayer`, you can *evaluate* the model, but you cannot continue to *train*
  it, since the "loss" method will raise.
  """

  def __init__(self,
               objective=None,
               constraints=None,
               placeholders=None,
               name=None,
               denominator_lower_bound=1e-3,
               loss_creator=lagrangian_optimizer.create_lagrangian_loss,
               **kwargs):
    """Creates a new `KerasLayer`.

    Since this is a Keras layer, the objective and constraints should presumably
    be constructed from contexts using `tfco.KerasPlaceholder`s to define their
    inputs.

    Args:
      objective: as in `tfco.RateContext'.
      constraints: as in `tfco.RateContext'.
      placeholders: collection of all `KerasPlaceholder` objects which serve as
        inputs to the "objective" and "constraints". Some of the inputs (but not
        all) can be found automatically, and this method will raise if any of
        these are absent from this list.
      name: as in `tf.keras.layers.Layer`.
      denominator_lower_bound: as in `tfco.RateContext'.
      loss_creator: optional, either `tfco.create_lagrangian_loss` (the
        default), or `tfco.create_proxy_lagrangian_loss`.
      **kwargs: additional arguments to pass to "loss_creator".
    """
    super(KerasLayer, self).__init__(name=name)

    if objective is None:
      # If we're being constructed using from_config(), then we'll do so in the
      # "empty" state, in which there are no variables, and no loss function,
      # but the layer is still a usable identity function for the purposes of
      # model evaluation.
      self._empty = True
    else:
      self._empty = False

      problem = rate_minimization_problem.RateMinimizationProblem(
          objective=objective,
          constraints=constraints,
          denominator_lower_bound=denominator_lower_bound,
          variable_fn=self._variable_fn)

      if placeholders is None:
        placeholders = []
      if not all(
          isinstance(placeholder, KerasPlaceholder)
          for placeholder in placeholders):
        raise TypeError("all placeholders passed to a KerasLayer must be "
                        "KerasPlaceholder objects")
      # We can't simply define self._placeholders to be the KerasPlaceholder
      # objects included in problem.inputs, since it would be possible to miss
      # some (e.g. arguments to the context.subset() method). Hence, we just
      # check that we were given all of the inputs that we know about.
      for placeholder in problem.inputs:
        if (isinstance(placeholder, KerasPlaceholder) and
            placeholder not in placeholders):
          raise ValueError("at least one input to the objective or "
                           "constraints was not excluded from the placeholder "
                           "list passed to the KerasLayer")
      # Make a copy of the placeholders list, to make sure that it doesn't
      # change after we leave this method.
      self._placeholders = list(placeholders)

      self._loss_fn, self._update_ops_fn, _ = loss_creator(
          minimization_problem=problem, variable_fn=self._variable_fn, **kwargs)

  def _variable_fn(self, initial_value, **kwargs):
    # We assert if self._empty is True, instead of raising, since this method
    # will only be called from the constructor, when self._empty is False.
    assert not self._empty
    shape = np.shape(initial_value)
    initializer = _KerasInitializer(initial_value)
    return self.add_weight(shape=shape, initializer=initializer, **kwargs)

  def loss(self, y_true, y_pred):
    """Loss function to pass to `Model.compile'.

    Args:
      y_true: float `Tensor`, the labels of the model (and, optionally,
        additional side information required by TFCO), as in
        `tf.keras.losses.Loss.__call__`.
      y_pred: float `Tensor`, the model's predictions, as in
        `tf.keras.losses.Loss.__call__`.

    Returns:
      A size-1 floating-point `Tensor`. Notice that, unlike
      `tf.keras.losses.Loss.__call__`, which returns a loss per each example in
      the batch, this loss performs the reduction internally, and returns only
      a *single* value.

    Raises:
      RuntimeError: if the class has been saved and loaded since it was
        constructed (which strips out the loss function, and other TFCO-specific
        information).
    """
    if self._empty:
      raise RuntimeError("after being saved+loaded, a tfco.KerasLayer is "
                         "empty, and in particular does not contain a valid "
                         "loss function--you can evaluate the model, but you "
                         "cannot further train it")
    for placeholder in self._placeholders:
      placeholder.assign(y_true, y_pred)
    with tf.control_dependencies(self._update_ops_fn()):
      result = self._loss_fn()
    for placeholder in self._placeholders:
      placeholder.assign()
    return result

  def get_config(self):
    # When we save a KerasLayer, we don't include *anything*, not even the
    # variables (although we could), since there is no way to save the loss
    # function (so the variables wouldn't be accessed, regardless). Instead,
    # when a saved KerasLayer is loaded, it does so in the "empty" state, in
    # which calls to loss() will raise.
    return {}

  # The default implementation of from_config is fine.


class KerasMetricWrapper(tf.keras.metrics.Metric):
  """Wraps a Keras `Metric`, accepting `KerasPlaceholder`s as inputs.

  This class both *is* a Keras `Metric`, and also *wraps* a Keras `Metric`
  instance. The purpose of this object is to make it easy to deal with standard
  Keras metrics in the setting in which e.g. the labels contain extra side
  information needed by TFCO (protected class memberships, predictions of a
  baseline model, etc.). A standard metric can't handle these extra label
  columns, so we need to extract the actual labels before calling the metric.

  To this end, this class is constructed with two parameters (among others):
  "predictions" and "labels", which are `KerasPlaceholder`s that tell us how to
  map the "y_true" and "y_pred" `Metric` parameters to the predictions and
  labels expected by the metric, respectively.
  """

  def __init__(self,
               wrapped,
               predictions=None,
               labels=None,
               from_logits=False,
               name=None,
               **kwargs):
    """Creates a new `KerasMetricWrapper`.

    The resulting object should act just like the `Metric` passed in the
    "wrapped" argument, except that its inputs will be extracted from "y_true"
    and "y_pred" as specified by the given "labels" and "predictions"
    `KerasPlaceholder`s (with the latter being possibly transformed due to the
    "from_logits" argument).

    Args:
      wrapped: Keras `Metric` instance to wrap.
      predictions: optional `KerasPlaceholder` object representing the
        predictions to pass to the wrapped `Metric`. If not provided, the
        "y_pred" parameter to "update_state" will pass through unchanged.
      labels: optional `KerasPlaceholder` object representing the labels to pass
        to the wrapped `Metric`. If not provided, the "y_true" parameter to
        "update_state" will pass through unchanged.
      from_logits: boolean, defaulting to `False`. If `True`, we will pass the
        predictions through a sigmoid before passing them on to the wrapped
        `Metric`. This is a workaround for the fact that some Keras `Metric`s do
        not support the "from_logits" parameter, themselves.
      name: string, the name of this `Metric`. If not specified, it will default
        to the name of the wrapped `Metric`.
      **kwargs: additional arguments to pass to the Keras `Metric` base class'
        constructor.
    """
    if predictions is not None and not isinstance(predictions,
                                                  KerasPlaceholder):
      raise TypeError("predictions parameter to KerasMetricWrapper must be a "
                      "KerasPlaceholder object")
    if labels is not None and not isinstance(labels, KerasPlaceholder):
      raise TypeError("labels parameter to KerasMetricWrapper must be a "
                      "KerasPlaceholder object")

    # If we're not given a name, we inherit from the wrapped Metric.
    if not name:
      name = wrapped.name
    super(KerasMetricWrapper, self).__init__(name=name, **kwargs)

    self._predictions = predictions
    self._labels = labels
    self._from_logits = from_logits
    self._wrapped = wrapped

  def update_state(self, y_true, y_pred, **kwargs):
    # Extract the labels from the KerasPlaceholder, if necessary.
    if self._labels:
      self._labels.assign(y_true, y_pred)
      labels = self._labels()
      self._labels.assign()
    else:
      labels = y_true

    # Extract the predictions from the KerasPlaceholder, if necessary.
    if self._predictions:
      self._predictions.assign(y_true, y_pred)
      predictions = self._predictions()
      self._predictions.assign()
    else:
      predictions = y_pred

    # Pass the predictions through a sigmoid, if necessary.
    if self._from_logits:
      predictions = tf.math.sigmoid(predictions)

    return self._wrapped.update_state(labels, predictions, **kwargs)

  def reset_states(self):
    return self._wrapped.reset_states()

  def result(self):
    return self._wrapped.result()
