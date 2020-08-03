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
import copy
import operator
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v2 as tf

from tensorflow_constrained_optimization.python.rates import defaults
from tensorflow_constrained_optimization.python.rates import helpers


class _DeferredTensorInput(helpers.RateObject):
  """Wrapper around allowed input types, supporting __hash__ and __eq__."""

  @classmethod
  def _list_to_tuple(cls, arg):
    """Recursively converts lists into tuples."""
    if not isinstance(arg, list):
      return arg
    return tuple(
        [_DeferredTensorInput._list_to_tuple(element) for element in arg])

  def __init__(self, value):
    """Creates a new `_DeferredTensorInput`.

    Args:
      value: either a `Tensor`-like object, or a nullary function returning such
        an object.
    """
    if isinstance(value, helpers.RateObject):
      raise TypeError(
          "a DeferredTensor may only be created from a Tensor-like object, "
          "or a nullary function returning such")
    self._value = value

    # For non-callable non-Tensor types, we make a deep copy to make
    # extra-certain that it is immutable (since we'll hash it).
    if not callable(self._value) and not tf.is_tensor(self._value):
      self._value = copy.deepcopy(self._value)

    # We memoize the hash, since it can be expensive to compute.
    self._hash = None

  @property
  def value(self):
    return self._value

  def __hash__(self):
    if self._hash is None:

      if callable(self._value):
        self._hash = hash(self._value)

      else:
        # We return the hash of a pair containing (i) the value, or None if
        # the value is a Tensor, and (ii) the id of the value, or None if the
        # value is not a Tensor.
        #
        # The reason for the fields being laid out like they are is that
        # __eq__() compares Tensors by id (using "is"), and non-Tensors by
        # value.
        if tf.is_tensor(self._value):
          identifier = (None, id(self._value))
        else:
          # We cast to a numpy array, then to a list (of lists of lists...), and
          # finally to a tuple (of tuples of tuples...) so that, even if we're
          # given a numpy type (which is not necessarily hashable), we'll wind
          # up with a hashable type.
          #
          # TODO: is there a more efficient way to do this? Notice that
          # we can't just hash a numpy array's storage, since arrays of
          # different types might be considered equal (e.g. [1.0, 2.0] ==
          # [1, 2]).
          identifier = (_DeferredTensorInput._list_to_tuple(
              np.array(self._value).tolist()), None)

        self._hash = hash(identifier)

    return self._hash

  def __eq__(self, other):
    if not isinstance(other, _DeferredTensorInput):
      return False
    if callable(self.value) != callable(other.value):
      return False

    if callable(self.value):
      # We can't actually determine the values without calling the callbacks, so
      # we only consider the states equal if they have the same callbacks.
      return self.value is other.value

    else:
      # If at least one of the objects is a Tensor, then we check that they're
      # the same object, instead of calling __eq__.
      #
      # In eager mode, we could potentially check for value-equality, by using
      # the np.array_equal() code below after *explicitly* casting the Tensors
      # to numpy arrays by calling Tensor.numpy(). This would probably be a bad
      # idea though, since if the Tensor is actually a variable, its value could
      # change in the future.
      if tf.is_tensor(self.value) or tf.is_tensor(other.value):
        return self.value is other.value

      # Every other allowed type can be handled by numpy.
      #
      # We can hope that in most cases, this will be quick (e.g. same object -->
      # equal, different shapes --> unequal), but if we're unlucky, this has the
      # potential to be slow.
      return np.array_equal(self.value, other.value)

  def __ne__(self, other):
    return not self.__eq__(other)


@six.add_metaclass(abc.ABCMeta)
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
  __call__ method takes two extra parameters: "structure_memoizer" and
  "value_memoizer" dicts that are used to memoize the TensorFlow `Variable`s
  associated with `DeferredVariable`s, and the values of certain
  `DeferredTensor`s, respectively). To make this class more easy-to-use than
  normal functions returning `Tensor`s, it also includes the `apply` function,
  which can be used to create new `DeferredTensor`s by performing operations on
  existing ones. Several standard operators are also overloaded to use `apply`.

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
  arg1 = ExplicitDeferredTensor(lambda: tensor1)
  arg2 = ExplicitDeferredTensor(tensor2, auto_cast=True)
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

  @abc.abstractmethod
  def _value_and_auto_cast(self, structure_memoizer, value_memoizer):
    """Returns the value of the `Tensor`, and whether it has no fixed type.

    The difference between the two memoizer parameters is that the
    "structure_memoizer" is responsible for memoizing quantities relating to the
    *structure* of the problem, whereas the "value_memoizer" remembers
    particular computed *values*. Whereas we have only one "structure_memoizer"
    per RateMinimizationProblem, we should be given a fresh "value_memoizer"
    every time the inputs change (i.e. at each iteration).

    Args:
      structure_memoizer: dict, which memoizes variables (among other things).
      value_memoizer: dict, which memoizes DeferredTensor values.

    Returns:
      A (`Tensor`-like, `Boolean`) pair containing the value of the
      `DeferredTensor`, and whether this value should be automatically
      type-promoted if necessary.
    """
    pass

  @abc.abstractproperty
  def inputs(self):
    """Returns the list of non-`DeferredVariable` dependencies.

    `DeferredTensor`s can be constructed from other `DeferredTensor`s via a call
    to apply(), as well as various operators (all of which thunk down to
    apply()). You can therefore think of a `DeferredTensor` as a parse tree of a
    python expression. The leaves of this tree will be `Tensor`-like objects,
    nullary functions returning `Tensor`-like objects, or `DeferredVariable`s.
    The apply() method keeps track of all dependencies included in the parse
    tree of a `DeferredTensor`, with the list of such `DeferredVariable`s being
    returned by the `variables` method, and the other two types of dependencies
    being returned by this one.

    Returns:
      A list of non-`DeferredVariable` dependencies.
    """

  @abc.abstractproperty
  def variables(self):
    """Returns the list of `DeferredVariable` dependencies.

    `DeferredTensor`s can be constructed from other `DeferredTensor`s via a call
    to apply(), as well as various operators (all of which thunk down to
    apply()). You can therefore think of a `DeferredTensor` as a parse tree of a
    python expression. Some of the leaves of the tree could be
    `DeferredVariable`s, which require initialization before they can be
    accessed. For this reason, apply() keeps track of all `DeferredVariable`s
    included in the parse tree of a `DeferredTensor`, with the list of such
    `DeferredVariable`s being returned by this method.

    Returns:
      A list of `DeferredVariable`s.
    """

  @abc.abstractmethod
  def __hash__(self):
    pass

  @abc.abstractmethod
  def __eq__(self, other):
    pass

  def __ne__(self, other):
    return not self.__eq__(other)

  def __call__(self, structure_memoizer, value_memoizer=None):
    """Returns the value of this `DeferredTensor`.

    This is the *only* place (aside from `DeferredVariable.update_ops`) that the
    value of the `DeferredTensor` will be accessed. In particular, if it was
    constructed from a nullary function, this is the only place that this
    function will be called.

    The difference between the two memoizer parameters is that the
    "structure_memoizer" is responsible for memoizing quantities relating to the
    *structure* of the problem, whereas the "value_memoizer" remembers
    particular computed *values*. Whereas we have only one "structure_memoizer"
    per RateMinimizationProblem, we should be given a fresh "value_memoizer"
    every time the inputs change (i.e. at each iteration).

    Args:
      structure_memoizer: dict, which memoizes variables (among other things).
      value_memoizer: dict, which memoizes DeferredTensor values.

    Returns:
      A `Tensor`-like object containing the value of this `DeferredTensor`.
    """
    value, _ = self._value_and_auto_cast(structure_memoizer, value_memoizer)
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
    if not callable(callback):
      raise TypeError("apply's callback argument must be callable")

    deferred_args = []
    for arg in args:
      if isinstance(arg, DeferredTensor):
        deferred_args.append(arg)
      elif not isinstance(arg, helpers.RateObject):
        deferred_args.append(ExplicitDeferredTensor(arg))
      else:
        raise TypeError("apply can only be called on DeferredTensor or "
                        "Tensor-like arguments")

    return _DerivedDeferredTensor(callback, deferred_args)

  def __neg__(self):
    return DeferredTensor.apply(operator.neg, self)

  # We need to override "both sides" of the below operators (e.g. both __add__
  # and __radd__) since other might not be a DeferredTensor (but will be casted
  # to one inside apply()).

  def __lt__(self, other):
    return DeferredTensor.apply(operator.lt, self, other)

  def __le__(self, other):
    return DeferredTensor.apply(operator.le, self, other)

  def __gt__(self, other):
    return DeferredTensor.apply(operator.gt, self, other)

  def __ge__(self, other):
    return DeferredTensor.apply(operator.ge, self, other)

  def __add__(self, other):
    return DeferredTensor.apply(operator.add, self, other)

  def __radd__(self, other):
    return DeferredTensor.apply(operator.add, other, self)

  def __sub__(self, other):
    return DeferredTensor.apply(operator.sub, self, other)

  def __rsub__(self, other):
    return DeferredTensor.apply(operator.sub, other, self)

  def __mul__(self, other):
    return DeferredTensor.apply(operator.mul, self, other)

  def __rmul__(self, other):
    return DeferredTensor.apply(operator.mul, other, self)

  def __truediv__(self, other):
    return DeferredTensor.apply(operator.truediv, self, other)

  def __rtruediv__(self, other):
    return DeferredTensor.apply(operator.truediv, other, self)

  def __getitem__(self, slice_spec):
    # Notice that because the slice_spec is a part of the callback, we can't use
    # operator.getitem, and therefore derived DeferredTensors created by
    # indexing the same DeferredTensor in the same way will not be considered
    # equal.
    return DeferredTensor.apply(lambda arg: arg[slice_spec], self)


class _DerivedDeferredTensor(DeferredTensor):
  """A `DeferredTensor` representing an expression.

  If we interpret a `DeferredTensor` as an expression tree, then
  `ExplicitDeferredTensor`s and `DeferredVariable`s will be the leaves, and
  `_DerivedDeferredTensor`s will be the internal nodes. Specifically, a
  `_DerivedDeferredTensor` represents an expression written in terms of other
  `DeferredTensor`s.
  """

  def __init__(self, callback, args):
    """Constructs a new `_DerivedDeferredTensor`.

    Args:
      callback: an n-ary function mapping `Tensor`-like objects to a
        `Tensor`-like object.
      args: list of `DeferredTensor`s to evaluate, and pass to the callback.

    Raises:
      TypeError: if value is neither a `Tensor`-like object nor a nullary
        function returning such an object.
    """
    super(_DerivedDeferredTensor, self).__init__()

    # We cast args to a tuple so that it's immutable, and can therefore be
    # hashed.
    self._callback = callback
    self._args = tuple(args)

    self._inputs = DeferredTensorInputList()
    for arg in self._args:
      self._inputs += arg.inputs

    self._variables = DeferredVariableList()
    for arg in self._args:
      self._variables += arg.variables

  def _value_and_auto_cast(self, structure_memoizer, value_memoizer):
    result = None

    key = (_DerivedDeferredTensor, self)
    if value_memoizer is not None and key in value_memoizer:
      result = value_memoizer[key]

    if result is None:
      values_and_auto_casts = [
          # pylint: disable=protected-access
          arg._value_and_auto_cast(structure_memoizer, value_memoizer)
          for arg in self._args
          # pylint: enable=protected-access
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
            for ii in xrange(len(self._args))
            if value_dtypes[ii] is not None and not auto_casts[ii])

        dtype = None
        if len(non_auto_cast_dtypes) == 1:
          # If we have exactly one non-auto-castable Tensor, then it determines
          # the dtype.
          dtype = non_auto_cast_dtypes.pop()
        elif not non_auto_cast_dtypes:
          # If we don't have any non-auto-castable Tensors, then the result of
          # this method should be auto-casted (ultimately, this will be the
          # second return value). In all other cases, it should not (the dtype
          # is fixed as that of the non-auto-castable Tensor).
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

      result = (self._callback(*values), auto_cast)

    if value_memoizer is not None and key not in value_memoizer:
      value_memoizer[key] = result

    return result

  @property
  def inputs(self):
    return self._inputs.list

  @property
  def variables(self):
    return self._variables.list

  def __hash__(self):
    return hash((self._callback, self._args))

  def __eq__(self, other):
    if not isinstance(other, _DerivedDeferredTensor):
      return False
    # pylint: disable=protected-access
    return self._callback == other._callback and self._args == other._args
    # pylint: enable=protected-access


class ExplicitDeferredTensor(DeferredTensor):
  """A `DeferredTensor` representing an explicit `Tensor`.

  If we interpret a `DeferredTensor` as an expression tree, then
  `ExplicitDeferredTensor`s and `DeferredVariable`s will be the leaves, and
  `_DerivedDeferredTensor`s will be the internal nodes. Unlike a
  `DeferredVariable`, which represents a TensorFlow variable, this class
  represents either a `Tensor`, or a nullary function returning such.
  """

  def __init__(self, value, auto_cast=False):
    """Constructs a new `ExplicitDeferredTensor`.

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
    super(ExplicitDeferredTensor, self).__init__()
    self._input = _DeferredTensorInput(value)
    self._auto_cast = auto_cast

  def _value_and_auto_cast(self, structure_memoizer, value_memoizer):
    del structure_memoizer

    if callable(self._input.value):
      result = None
      if value_memoizer is not None:
        key = (ExplicitDeferredTensor, self)
        if key not in value_memoizer:
          value_memoizer[key] = (self._input.value(), self._auto_cast)
        result = value_memoizer[key]
      else:
        result = (self._input.value(), self._auto_cast)
    else:
      del value_memoizer
      result = self._input.value, self._auto_cast

    return result

  @property
  def inputs(self):
    # The only input upon which this ExplicitDeferredTensor depends is its own.
    return [self._input.value]

  @property
  def variables(self):
    # ExplicitDeferredTensors are non-variable leaves, and therefore have no
    # dependencies on DeferredVariables.
    return []

  def __hash__(self):
    return hash((self._input, self._auto_cast))

  def __eq__(self, other):
    if not isinstance(other, helpers.RateObject):
      other = ExplicitDeferredTensor(other)
    elif not isinstance(other, ExplicitDeferredTensor):
      return False
    return (self._input.__eq__(other._input) and
            self._auto_cast == other._auto_cast)


class DeferredVariable(DeferredTensor):
  """A `DeferredTensor` representing a variable.

  If we interpret a `DeferredTensor` as an expression tree, then
  `ExplicitDeferredTensor`s and `DeferredVariable`s will be the leaves, and
  `_DerivedDeferredTensor`s will be the internal nodes. Unlike an
  `ExplicitDeferredTensor`, which wraps a normal `Tensor` (or a nullary function
  returning such), this class is intended as a substitute for `tf.Variable`.

  Unlike a normal `tf.Variable`, in this library we want all variables to be
  owned by a `RateMinimizationProblem`. Furthermore, we want there to be
  multiple copies of the "same" variable, if there are multiple
  `RateMinimizationProblem`s.

  The storage for all `DeferredVariables` must be *explicitly* created by
  calling `create` (this will happen inside the `RateMinimizationProblem`
  constructor). The resulting variable will be stored in the
  "structure_memoizer" dict owned by the `RateMinimizationProblem`. After it's
  created, the value can be accessed as usual for a `DeferredTensor`: by calling
  it (`DeferredTensor`'s __call__ method).
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
      update_ops_fn: an optional function taking three parameters: the
        `tf.Variable` corresponding to this `DeferredVariable`, a
        "structure_memoizer" dict, and a "value_memoizer" dict; it returns a
        list of ops that should be executed before each training iteration.
      auto_cast: if `True`, then calls to `DeferredTensor.apply` will attempt to
        perform automatic type promotion on the resulting `DeferredVariable`.
    """
    super(DeferredVariable, self).__init__()

    self._initial_value = initial_value
    self._trainable = trainable
    self._name = name
    self._dtype = dtype
    self._constraint = constraint
    self._update_ops_fn = update_ops_fn
    self._auto_cast = auto_cast

  def _value_and_auto_cast(self, structure_memoizer, value_memoizer):
    del value_memoizer
    key = (DeferredVariable, self)
    if key not in structure_memoizer:
      raise RuntimeError("attempted to read a DeferredVariable that has not "
                         "yet been created")
    return structure_memoizer[key], self._auto_cast

  @property
  def inputs(self):
    # DeferredVariables are excluded from the list of inputs (they are accessed
    # via the "variables" method).
    return []

  @property
  def variables(self):
    # The only DeferredVariable upon which this DeferredVariable depends is
    # itself.
    return [self]

  def __hash__(self):
    return hash(id(self))

  def __eq__(self, other):
    return self is other

  def create(self, structure_memoizer):
    """Creates the variable owned by this `DeferredVariable`."""
    key = (DeferredVariable, self)
    if key in structure_memoizer:
      raise RuntimeError("attempted to create a DeferredVariable that has "
                         "already been created")
    structure_memoizer[key] = structure_memoizer[defaults.VARIABLE_FN_KEY](
        initial_value=self._initial_value,
        trainable=self._trainable,
        name=self._name,
        dtype=self._dtype,
        constraint=self._constraint)

  def update_ops(self, structure_memoizer, value_memoizer=None):
    """Creates and returns a list of ops to run at the start of train_op.

    Aside from `DeferredTensor`'s __call__ method, this is the only place that
    the value of the `DeferredVariable` will be accessed.

    The difference between the two memoizer parameters is that the
    "structure_memoizer" is responsible for memoizing quantities relating to the
    *structure* of the problem, whereas the "value_memoizer" remembers
    particular computed *values*. Whereas we have only one "structure_memoizer"
    per RateMinimizationProblem, we should be given a fresh "value_memoizer"
    every time the inputs change (i.e. at each iteration).

    Args:
      structure_memoizer: dict, which memoizes variables (among other things).
      value_memoizer: dict, which memoizes DeferredTensor values.

    Returns:
      A list of ops.
    """
    if self._update_ops_fn is None:
      return []
    return self._update_ops_fn(
        self.__call__(structure_memoizer, value_memoizer), structure_memoizer,
        value_memoizer)


class DeferredTensorInputList(helpers.RateObject):
  """Represents a list of `DeferredTensor` inputs.

  Aside from having a very stripped-down interface compared to a normal Python
  list, this class also differs in that (i) it verifies that every element it
  contains is a valid `DeferredTensor` input, and (ii) duplicate elements are
  removed (but, unlike a set, order is preserved).
  """

  def __init__(self, collection=None):
    """Creates a new `DeferredTensorInputList` from a collection."""
    # We need to wrap _DeferredTensorInput around every element so that we have
    # __hash__ and __eq__, allowing us to use UniqueList (which is basically the
    # entire point of this class).
    if collection is not None:
      collection = [_DeferredTensorInput(element) for element in collection]
    self._list = helpers.UniqueList(
        collection=collection, element_type=_DeferredTensorInput)

  @property
  def list(self):
    """Returns the contents as a Python list."""
    return [element.value for element in self._list]

  def append(self, element):
    """Appends a new element to the list, ignoring duplicates."""
    self._list.append(_DeferredTensorInput(element))

  def __eq__(self, other):
    if not isinstance(other, DeferredTensorInputList):
      return False
    return self._list.__eq__(other._list)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __len__(self):
    """Returns the length of this `DeferredTensorInputList`."""
    return self._list.__len__()

  def __iter__(self):
    """Returns an iterator over a COPY of the wrapped list."""
    # We call "list" instead of "_list" since we want the inputs themselves, not
    # the _DeferredTensorInput wrappers. Notice that this copies the list.
    return self.list.__iter__()

  def __add__(self, other):
    """Appends two `DeferredTensorInputList`s."""
    result = DeferredTensorInputList(self)
    for element in other:
      result.append(element)
    return result

  def __radd__(self, other):
    """Appends two `DeferredTensorInputList`s."""
    result = DeferredTensorInputList(other)
    for element in self:
      result.append(element)
    return result

  def __getitem__(self, slice_spec):
    """Returns a single element or a slice of this `DeferredTensorInputList`."""
    # We call "list" instead of "_list" since we want the inputs themselves, not
    # the _DeferredTensorInput wrappers. Notice that this copies the list.
    result = self.list[slice_spec]
    if isinstance(result, list):
      result = DeferredTensorInputList(result)
    return result


class DeferredVariableList(helpers.UniqueList):
  """Represents a list of `DeferredVariable`s.

  Aside from having a very stripped-down interface compared to a normal Python
  list, this class also differs in that (i) it verifies that every element it
  contains is a `DeferredVariable` object, and (ii) duplicate elements are
  removed (but, unlike a set, order is preserved).
  """

  def __init__(self, collection=None):
    """Creates a new `DeferredVariableList` from a collection."""
    super(DeferredVariableList, self).__init__(
        collection=collection, element_type=DeferredVariable)
