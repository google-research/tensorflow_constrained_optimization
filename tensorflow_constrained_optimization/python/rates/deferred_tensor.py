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
"""Contains the `DeferredTensor` class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow_constrained_optimization.python.rates import helpers


@six.add_metaclass(abc.ABCMeta)
class _DeferredTensorState(helpers.RateObject):
  """Base class for internal state of a `DeferredTensor`."""

  @abc.abstractmethod
  def value_and_auto_cast(self, memoizer):
    """Returns the value of the `Tensor`, and whether it has no fixed type.

    Args:
      memoizer: dict, which memoizes variables (among other things).

    Returns:
      A (`Tensor`-like, `Boolean`) pair containing the value of the
      `DeferredTensor`, and whether this value should be automatically
      type-promoted if necessary.
    """
    pass

  @abc.abstractmethod
  def __hash__(self):
    pass

  @abc.abstractmethod
  def __eq__(self, other):
    pass

  def __ne__(self, other):
    return not self.__eq__(other)


class _StaticDeferredTensorState(_DeferredTensorState):
  """Internal state for a `DeferredTensor` with known value."""

  def __init__(self, value, auto_cast):
    """Creates a new `_StaticDeferredTensorState`.

    Args:
      value: `Tensor`-like, the value of the `DeferredTensor`.
      auto_cast: `Boolean`, whether the value should be automatically
        type-promoted, if necessary. Only applies if "value" is a `Tensor`:
          non-`Tensor` types are always auto-castable.
    """
    assert not callable(value)
    self._value = value
    self._auto_cast = auto_cast

  @property
  def value(self):
    return self._value

  @property
  def auto_cast(self):
    return self._auto_cast

  def value_and_auto_cast(self, memoizer):
    del memoizer
    return self._value, self._auto_cast

  def __hash__(self):
    return hash((self._value, self._auto_cast))

  def __eq__(self, other):
    if not isinstance(other, _StaticDeferredTensorState):
      return False
    if self.auto_cast != other.auto_cast:
      return False

    # If at least one of the objects is a Tensor, then we check that they're the
    # same object, instead of calling __eq__.
    #
    # NOTE: in eager mode, we could potentially check for value-equality, using
    # the np.array_equal() code below. However, when we did this, it didn't seem
    # to necessarily work when the Tensors in question were actually
    # tf.Variables (it worked in TensorFlow 2.0 and lower, but not 2.1).
    if tf.is_tensor(self.value) or tf.is_tensor(other.value):
      return self.value is other.value

    # Every other allowed type can be handled by numpy.
    #
    # We can hope that in most cases, this will be quick (e.g. same object -->
    # equal, different shapes --> unequal), but if we're unlucky, this has the
    # potential to be slow.
    return np.array_equal(self.value, other.value)


class _CallableDeferredTensorState(_DeferredTensorState):
  """Internal state for a `DeferredTensor` with unknown value.

  The value isn't exactly "unknown". Rather, it isn't immediately available,
  since it is returned by the callback function passed to the constructor, which
  should not be called until we're inside a `RateMinimizationProblem`.
  """

  def __init__(self, callback, auto_cast):
    """Creates a new `_CallableDeferredTensorState`.

    Args:
      callback: nullary function returning a `Tensor`-like, the value of the
        `DeferredTensor`.
      auto_cast: `Boolean`, whether the value should be automatically
        type-promoted, if necessary. Only applies if "callback" returns a
        `Tensor`: non-`Tensor` types are always auto-castable.
    """
    assert callable(callback)
    self._callback = callback
    self._auto_cast = auto_cast

  @property
  def callback(self):
    return self._callback

  @property
  def auto_cast(self):
    return self._auto_cast

  def value_and_auto_cast(self, memoizer):
    del memoizer
    return self._callback(), self._auto_cast

  def __hash__(self):
    return hash((self._callback, self._auto_cast))

  def __eq__(self, other):
    if not isinstance(other, _CallableDeferredTensorState):
      return False
    # We can't actually determine the values without calling the callbacks, so
    # we only consider the states equal if they have the same callbacks.
    return (self.callback is other.callback and
            self.auto_cast == other.auto_cast)


class _DerivedDeferredTensorState(_DeferredTensorState):
  """Internal state for a `DeferredTensor` derived from others.

  This `DeferredTensor` state will be used in two cases: (i) when a new
  `DeferredTensor` is constructed from others via a call to
  `DeferredTensor.apply`(), or (ii) when creating a `DeferredVariable`.
  """

  def __init__(self, callback):
    """Creates a new `_DerivedDeferredTensorState`.

    Args:
      callback: nullary function returning a (`Tensor`-like, `Boolean`) pair,
        the first element of which is the `DeferredTensor`, with the second
        indicating whether the value should be automatically type-promoted, if
        necessary.
    """
    assert callable(callback)
    self._callback = callback

  @property
  def callback(self):
    return self._callback

  def value_and_auto_cast(self, memoizer):
    return self._callback(memoizer)

  def __hash__(self):
    return hash(self._callback)

  def __eq__(self, other):
    if not isinstance(other, _DerivedDeferredTensorState):
      return False
    # We can't actually determine the values without calling the callbacks, so
    # we only consider the states equal if they have the same callbacks.
    return self.callback is other.callback


class DeferredTensor(helpers.RateObject):
  """Wrapper around `Tensor`-like objects, and functions returning such.

  In graph mode, a `Tensor` is an edge in the graph, so it's possible to
  differentiate through it by walking along the incoming edges. In eager mode, a
  `Tensor` is little more than an array, so if you want to differentiate through
  it, then you need to have some additional knowledge of how the `Tensor` was
  computed. For this reason, in eager mode, it's common to pass nullary
  functions that compute and return `Tensor`s, instead of `Tensor`s themselves,
  when you want them to be involved in automatic differentiation.

  This class is essentially a more-convenient representation of such functions.
  The idea is that a `DeferredTensor` should behave like a function returning a
  `Tensor`, but that it should be easy to use--in particular, it should be
  unnecessary, most of the time, to create new `DeferredTensor` by writing a
  function that returns a `Tensor`, in which other
  functions-that-return-`Tensor`s are called. To look like a function returning
  a `Tensor`, instances of this class are evaluated by calling them (but the
  __call__ method takes one extra parameter: a "memoizer" dict that is used to
  store the TensorFlow `Variable`s associated with `DeferredVariable`s). To make
  this class more easy-to-use than normal functions returning `Tensor`s, it also
  includes the `apply` function, which can be used to create new
  `DeferredTensor`s by performing operations on existing ones. Several standard
  operators are also overloaded to use `apply`.

  The most important property of the `apply` function is that it creates a
  `DeferredTensor` representing a *closure*, i.e. a function that computes the
  desired quantity, instead of computing the function application immediately.
  For this reason, this class works fine with automatic differentiation, even in
  eager mode.

  The second-most important property of the `apply` function is that it
  (optionally) supports automatic type promotion for `Tensor`s. This is
  necessary since, when all we have is a function returning a `Tensor`, and not
  a `Tensor` itself, there is no way to determine the dtype of the resulting
  `Tensor` without calling the function. So, creating a new `DeferredTensor`
  with the same dtype as an existing `DeferredTensor` isn't really possible.
  Instead, we can create a new `DeferredTensor` with a potentially-different
  dtype, and then flag it as auto-castable.

  For example, imagine that "tensor1" is a float32 `Tensor`, and "tensor2" is an
  int16 `Tensor`. Consider the following code:

  ```python
  arg1 = DeferredTensor(lambda: tensor1)
  arg2 = DeferredTensor(tensor2, auto_cast=True)
  result = DeferredTensor.apply(tf.maximum, arg1, arg2)
  result_tensor = result()
  ```

  Here, we begin by constructing two `DeferredTensor`s, the first of which
  ("arg1") is created from a nullary function returning "tensor1", and the
  second of which ("arg2") will participate in automatic type promotion
  (`auto_cast=True`). The two arguments are then passed into the `tf.maximum`
  function using `DeferredTensor.apply`, which will first cast "arg2" to have
  dtype float32. The resulting `DeferredTensor`, "result", is finally converted
  into a normal `Tensor` by calling it.
  """

  def __init__(self, value, auto_cast=False):
    """Constructs a new `DeferredTensor.

    Args:
      value: either a `Tensor`-like object, or a nullary function returning such
        an object.
      auto_cast: if `True`, then calls to `apply` will attempt to perform
        automatic type promotion on the resulting `DeferredTensor`. Only applies
        if "value" is a `Tensor` or a function returning a `Tensor`:
          non-`Tensor` types are always auto-castable.

    Raises:
      TypeError: if value is neither a `Tensor`-like object nor a nullary
        function returning such an object.
    """
    if isinstance(value, _DeferredTensorState):
      # We permit a DeferredTensor to be created from a _DeferredTensorState,
      # but this is for internal use *only* (it's used to create a
      # DeferredTensor with a _DerivedDeferredTensorState). Notice that the
      # auto_cast parameter is ignored (since it's a part of the state).
      del auto_cast
      self._state = value
    elif isinstance(value, helpers.RateObject):
      raise TypeError("a DeferredTensor may only be created from a Tensor-like "
                      "object, or a nullary function returning such")
    elif callable(value):
      # If we're given a callable value, then we treat it as a nullary function
      # returning a Tensor-like object.
      self._state = _CallableDeferredTensorState(value, auto_cast)
    else:
      # If we're not given a callable value, then we treat it as a Tensor-like
      # object.
      self._state = _StaticDeferredTensorState(value, auto_cast)

  def __call__(self, memoizer):
    """Returns the value of this `DeferredTensor`.

    This is the *only* place (aside from `DeferredVariable.update_ops`) that the
    value of the `DeferredTensor` will be accessed. In particular, if it was
    constructed from a nullary function, this is the only place that this
    function will be called.

    Args:
      memoizer: dict, which memoizes variables (among other things).

    Returns:
      A `Tensor`-like object containing the value of this `DeferredTensor`.
    """
    value, _ = self._state.value_and_auto_cast(memoizer)
    return value

  @classmethod
  def apply(cls, callback, *args):
    """Returns a new `DeferredTensor` representing callback(*args).

    When you want to apply a function to one or more `DeferredTensor`s, it would
    be most natural to extract the value from each of them by calling it, then
    call the function that you wish to apply on the results, and finally wrap
    the result in a new `DeferredTensor`. This, however, would cause the input
    `DeferredTensor`s to be called someplace where you might not want them to be
    called: the entire point of this class is to make sure that __call__ is only
    used inside the scope of a `RateMinimizationProblem`.

    Hence, this function exists to create a closure representing the function
    application, and then wrapping the resulting closure in a new
    `DeferredTensor`. The closure, and therefore the calls to the
    `DeferredTensor`s that it contains, will not itself be called until __call__
    is used on the `DeferredTensor` returned by this function.

    In addition, this is the (only) place where the "auto_cast" constructor
    parameter is handled. Any `DeferredTensor` for which "auto_cast" is `True`
    will participate in automatic type promotion: essentially, this function
    will attempt to ensure that all arguments passed to the callback have the
    same dtype, by casting any auto-castable `Tensor`s.

    Args:
      callback: an n-ary function mapping `Tensor`-like objects to a
        `Tensor`-like object.
      *args: the `DeferredTensor`s or `Tensor`-like objects to evaluate, and
        pass to the callback.

    Returns:
      A new `DeferredTensor` representing the requested function application.
      This is an *abstract* representation: the function application will not
      actually be *performed* until one calls the result.

    Raises:
      TypeError: if any of the arguments are not `DeferredTensor`s or
        `Tensor`-like objects.
    """
    deferred_args = []
    for arg in args:
      if isinstance(arg, DeferredTensor):
        deferred_args.append(arg)
      elif not isinstance(arg, helpers.RateObject):
        deferred_args.append(DeferredTensor(arg))
      else:
        raise TypeError("apply can only be called on DeferredTensor or "
                        "Tensor-like arguments")

    states = [
        arg._state  # pylint: disable=protected-access
        for arg in deferred_args
    ]

    def value_and_auto_cast_fn(memoizer):
      """Returns the application value, and whether it should be auto-casted."""
      values_and_auto_casts = [
          state.value_and_auto_cast(memoizer) for state in states
      ]
      values = [value for value, _ in values_and_auto_casts]
      # The auto_cast flag is only meaningful for Tensors. For all other types,
      # we take auto_cast to be False by convention (but don't be misled: all
      # non-Tensor types will undergo type promotion automatically).
      auto_casts = [
          tf.is_tensor(value) and auto_cast
          for value, auto_cast in values_and_auto_casts
      ]

      # If any of the input arguments are auto-castable `Tensor`s, then we have
      # to figure out what type we should cast them to. Otherwise, we don't need
      # to bother with all of this, and can set the auto_cast flag (for the
      # result) to False (this is safe: observe that all non-Tensor types are
      # automatically auto-castable, regardless of the value of the auto_cast
      # flag).
      auto_cast = False
      if any(auto_casts):
        # Get the dtype of every Tensor, and take the dtype of every non-Tensor
        # to be None (for python scalars and numpy objects, type promotion will
        # be performed automatically, so the only types that "matter" are those
        # of the Tensors).
        value_dtypes = []
        for value in values:
          if tf.is_tensor(value):
            value_dtypes.append(value.dtype.base_dtype)
          else:
            value_dtypes.append(None)

        # Here, we'll fill in "non_auto_cast_dtypes" with the dtypes of
        # non-auto-castable Tensors. If all non-auto-castable Tensors have the
        # same dtype, then we'll cast every auto-castable Tensor to have this
        # dtype. If there are non-auto-castable Tensors with different dtypes,
        # then we won't perform any casting at all.
        non_auto_cast_dtypes = set(
            value_dtypes[ii]
            for ii in xrange(len(states))
            if value_dtypes[ii] is not None and not auto_casts[ii])

        dtype = None
        if len(non_auto_cast_dtypes) == 1:
          # If we have exactly one non-auto-castable Tensor, then it determines
          # the dtype.
          dtype = non_auto_cast_dtypes.pop()
        elif not non_auto_cast_dtypes:
          # If we don't have any non-auto-castable Tensors, then the result of
          # this method should be auto-casted (ultimately, this will be the
          # second return value). In all other cases, it should not.
          auto_cast = True

          # Since every input argument is either an auto-castable Tensor, or a
          # non-Tensor, we do not have a hard dtype requirement, so we'll use
          # numpy's type promotion rules to figure out the dtype.
          numpy_dtypes = [
              value_dtype for value_dtype in value_dtypes
              if value_dtype is not None
          ]
          if not all(
              numpy_dtype.is_numpy_compatible for numpy_dtype in numpy_dtypes):
            raise ValueError("auto-castable DeferredTensors must use numpy "
                             "compatible dtypes")
          numpy_dtypes = [
              numpy_dtype.as_numpy_dtype for numpy_dtype in numpy_dtypes
          ]
          if numpy_dtypes:
            dtype = tf.as_dtype(np.result_type(*numpy_dtypes))

        if dtype is not None:
          for ii in xrange(len(values_and_auto_casts)):
            value_dtype = value_dtypes[ii]
            if (auto_casts[ii] and value_dtype is not None and
                value_dtype != dtype):
              values[ii] = tf.cast(values[ii], dtype=dtype)

      return callback(*values), auto_cast

    # FUTURE WORK: if all of the inputs are static, then the memoizer will be
    # ignored, and we could just construct a new static DeferredTensor here.
    return DeferredTensor(_DerivedDeferredTensorState(value_and_auto_cast_fn))

  def __hash__(self):
    return self._state.__hash__()

  def __eq__(self, other):
    if not isinstance(other, helpers.RateObject):
      other = DeferredTensor(other)
    elif not isinstance(other, DeferredTensor):
      return NotImplemented
    return self._state == other._state  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self.__eq__(other)

  def __neg__(self):
    return DeferredTensor.apply(lambda arg: -arg, self)

  # We need to override "both sides" of the below operators (e.g. both __add__
  # and __radd__) since other might not be a DeferredTensor (but will be casted
  # to one inside apply()).

  def __lt__(self, other):
    return DeferredTensor.apply(lambda left, right: left < right, self, other)

  def __le__(self, other):
    return DeferredTensor.apply(lambda left, right: left <= right, self, other)

  def __gt__(self, other):
    return DeferredTensor.apply(lambda left, right: left > right, self, other)

  def __ge__(self, other):
    return DeferredTensor.apply(lambda left, right: left >= right, self, other)

  def __add__(self, other):
    return DeferredTensor.apply(lambda left, right: left + right, self, other)

  def __radd__(self, other):
    return DeferredTensor.apply(lambda left, right: left + right, other, self)

  def __sub__(self, other):
    return DeferredTensor.apply(lambda left, right: left - right, self, other)

  def __rsub__(self, other):
    return DeferredTensor.apply(lambda left, right: left - right, other, self)

  def __mul__(self, other):
    return DeferredTensor.apply(lambda left, right: left * right, self, other)

  def __rmul__(self, other):
    return DeferredTensor.apply(lambda left, right: left * right, other, self)

  def __truediv__(self, other):
    return DeferredTensor.apply(lambda left, right: left / right, self, other)

  def __rtruediv__(self, other):
    return DeferredTensor.apply(lambda left, right: left / right, other, self)

  def __getitem__(self, slice_spec):
    return DeferredTensor.apply(lambda arg: arg[slice_spec], self)


class DeferredVariable(DeferredTensor):
  """A `DeferredTensor` representing a variable.

  This class is intended as a substitute for `tf.Variable`, with the main
  difference being that we want all variables to be owned by a
  `RateMinimizationProblem`. Furthermore, we want there to be multiple copies of
  the "same" variable, if there are multiple `RateMinimizationProblem`s.

  The storage for all `DeferredVariables` must be *explicitly* created by
  calling `create` (this will happen inside the `RateMinimizationProblem`
  constructor). The resulting `tf.Variable` will be stored in the "memoizer"
  dict owned by the `RateMinimizationProblem`. After it's created, the value can
  be accessed as usual for a `DeferredTensor`: by calling it (`DeferredTensor`'s
  __call__ method).
  """

  def __init__(self,
               initial_value,
               trainable=None,
               name=None,
               dtype=None,
               constraint=None,
               update_ops_fn=None,
               auto_cast=False):
    """Constructs a new `DeferredVariable`.

    Args:
      initial_value: as in `tf.Variable`'s constructor.
      trainable: as in `tf.Variable`'s constructor.
      name: as in `tf.Variable`'s constructor.
      dtype: as in `tf.Variable`'s constructor.
      constraint: as in `tf.Variable`'s constructor.
      update_ops_fn: an optional function taking the two parameters: the
        `tf.Variable` corresponding to this `DeferredVariable`, and a "memoizer"
        dict; and returning a list of ops that should be executed before each
        training iteration.
      auto_cast: if `True`, then calls to `DeferredTensor.apply` will attempt to
        perform automatic type promotion on the resulting `DeferredVariable`.
    """
    self._initial_value = initial_value
    self._trainable = trainable
    self._name = name
    self._dtype = dtype
    self._constraint = constraint

    def value_and_auto_cast_fn(memoizer):
      """Returns the variable's value, and whether it should be auto-casted."""
      key = (DeferredVariable, self)
      if key not in memoizer:
        raise RuntimeError("attempted to read a DeferredVariable that has not "
                           "yet been created")
      return memoizer[key], auto_cast

    self._update_ops_fn = update_ops_fn
    super(DeferredVariable,
          self).__init__(_DerivedDeferredTensorState(value_and_auto_cast_fn))

  def create(self, memoizer):
    """Creates the `tf.Variable` owned by this `DeferredVariable`."""
    key = (DeferredVariable, self)
    if key in memoizer:
      raise RuntimeError("attempted to create a DeferredVariable that has "
                         "already been created")
    memoizer[key] = tf.compat.v2.Variable(
        initial_value=self._initial_value,
        trainable=self._trainable,
        name=self._name,
        dtype=self._dtype,
        constraint=self._constraint)

  def update_ops(self, memoizer):
    """Creates and returns a list of ops to run at the start of train_op.

    Aside from `DeferredTensor`'s __call__ method, this is the only place that
    the value of the `DeferredVariable` will be accessed.

    Args:
      memoizer: dict, which memoizes variables (among other things).

    Returns:
      A list of ops.
    """
    if self._update_ops_fn is None:
      return []
    return self._update_ops_fn(self.__call__(memoizer), memoizer)
