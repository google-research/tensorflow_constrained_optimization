<!-- Copyright 2018 The TensorFlow Constrained Optimization Authors. All Rights
Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
=============================================================================-->

# TensorFlow Constrained Optimization (TFCO)

TFCO is a library for optimizing inequality-constrained problems in TensorFlow.
In the most general case, both the objective function and the constraints are
represented as `Tensor`s, giving users the maximum amount of flexibility in
specifying their optimization problems. Constructing these `Tensor`s can be
cumbersome, so we also provide helper functions to make it easy to construct
constrained optimization problems based on *rates*, i.e. proportions of the
training data on which some event occurs (e.g. the error rate, true positive
rate, recall, etc).

For full details, motivation, and theoretical results on the approach taken by
this library, please refer to:

> Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
> Constrained Optimization".
> [ALT'19](http://alt2019.algorithmiclearningtheory.org/).
> [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

which will be referred to as [CoJiSr19] throughout the remainder of this
document, and:

> Cotter, Gupta, Jiang, Srebro, Sridharan, Wang, Woodworth and You. "Training
> Well-Generalizing Classifiers for Fairness Metrics and Other Data-Dependent
> Constraints".
> [https://arxiv.org/abs/1807.00028](https://arxiv.org/abs/1807.00028)

which will be referred to as [CotterEtAl18].

### Proxy constraints

Imagine that we want to constrain the recall of a binary classifier to be at
least 90%. Since the recall is proportional to the number of true positive
classifications, which itself is a sum of indicator functions, this constraint
is non-differentiable, and therefore cannot be used in a problem that will be
optimized using a (stochastic) gradient-based algorithm.

For this and similar problems, TFCO supports so-called *proxy constraints*,
which are differentiable (or sub/super-differentiable) approximations of the
original constraints. For example, one could create a proxy recall function by
replacing the indicator functions with sigmoids. During optimization, each proxy
constraint function will be penalized, with the magnitude of the penalty being
chosen to satisfy the corresponding *original* (non-proxy) constraint.

On a problem including proxy constraints&mdash;even a convex problem&mdash;the
Lagrangian approach discussed above isn't guaranteed to work. However, a
different algorithm, based on minimizing *swap regret* on a slightly *different*
formulation&mdash;which we call the "proxy-Lagrangian" formulation&mdash;does
work.

### Shrinking

This library is designed to deal with a very flexible class of constrained
problems, but this flexibility makes optimization considerably more difficult:
on a non-convex problem, if one uses the "standard" approach of introducing a
Lagrange multiplier for each constraint, and then jointly maximizing over the
Lagrange multipliers and minimizing over the model parameters, then a stable
stationary point might not even *exist*. Hence, in some cases, oscillation,
instead of convergence, is inevitable.

Thankfully, it turns out that even if, over the course of optimization, no
*particular* iterate does a good job of minimizing the objective while
satisfying the constraints, the *sequence* of iterates, on average, usually
will. This observation suggests the following approach: at training time, we'll
periodically snapshot the model state during optimization; then, at evaluation
time, each time we're given a new example to evaluate, we'll sample one of the
saved snapshots uniformly at random, and apply it to the example. This
*stochastic model* will generally perform well, both with respect to the
objective function, and the constraints.

In fact, we can do better: it's possible to post-process the set of snapshots to
find a distribution over at most **m+1** snapshots, where **m** is the number of
constraints, that will be at least as good (and will usually be much better)
than the (much larger) uniform distribution described above. If you're unable or
unwilling to use a stochastic model at all, then you can instead use a heuristic
to choose the single best snapshot.

In many cases, these issues can be ignored. However, if you experience issues
with oscillation during training, or if you want to squeeze every last drop of
performance out of your model, consider using the "shrinking" procedure of
[CoJiSr19], which is implemented in the "candidates.py" file.

## Public contents

*   [constrained_minimization_problem.py](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/constrained_minimization_problem.py):
    contains the `ConstrainedMinimizationProblem` interface, representing an
    inequality-constrained problem. Your own constrained optimization problems
    should be represented using implementations of this interface. If using the
    rate-based helpers, such objects can be constructed as
    `RateMinimizationProblem`s.

*   [candidates.py](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/candidates.py):
    contains two functions, `find_best_candidate_distribution` and
    `find_best_candidate_index`. Both of these functions are given a set of
    candidate solutions to a constrained optimization problem, from which the
    former finds the best distribution over at most **m+1** candidates, and the
    latter heuristically finds the single best candidate. As discussed above,
    the set of candidates will typically be model snapshots saved periodically
    during optimization. Both of these functions require that scipy be
    installed.

    The `find_best_candidate_distribution` function implements the approach
    described in Lemma 3 of [CoJiSr19], while `find_best_candidate_index`
    implements the heuristic used for hyperparameter search in the experiments
    of Section 5.2.

*   [Optimizing general inequality-constrained problems](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/train/)

    *   [constrained_optimizer.py](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/train/constrained_optimizer.py):
        contains the `ConstrainedOptimizer` interface, which is similar to (but
        different from) `tf.train.Optimizer`, with the main difference being
        that `ConstrainedOptimizer`s are given `ConstrainedMinimizationProblem`s
        to optimize, and perform constrained optimization.

    *   [lagrangian_optimizer.py](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/train/lagrangian_optimizer.py):
        contains the `LagrangianOptimizer` implementation, which is a
        `ConstrainedOptimizer` implementing the Lagrangian approach discussed
        above (with additive updates to the Lagrange multipliers). You should
        use this optimizer for problems *without* proxy constraints. It may also
        work for problems with proxy constraints, but we recommend using a
        proxy-Lagrangian optimizer, instead.

        This optimizer is most similar to Algorithm 3 in Appendix C.3 of
        [CoJiSr19], and is discussed in Section 3. The two differences are that
        it uses proxy constraints (if they're provided) in the update of the
        model parameters, and uses `tf.train.Optimizer`s, instead of SGD, for
        the "inner" updates.

    *   [proxy_lagrangian_optimizer.py](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/train/proxy_lagrangian_optimizer.py):
        contains the `ProxyLagrangianOptimizer` implementation, which is a
        `ConstrainedOptimizer` implementing the proxy-Lagrangian approach
        mentioned above. We recommend using this optimizer for problems *with*
        proxy constraints.

        The `ProxyLagrangianOptimizer` with multiplicative swap-regret updates
        is most similar to Algorithm 2 in Section 4 of [CoJiSr19], with the
        difference being that it uses `tf.train.Optimizer`s, instead of SGD, for
        the "inner" updates.

*   [Helpers for constructing rate-based optimization problems](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/rates/)

    *   [subsettable_context.py](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/rates/subsettable_context.py):
        contains the `rate_context` method, which takes a `Tensor` of
        predictions (i.e. the output of a TensorFlow model, through which
        gradients can be propagated), and optionally `Tensor`s of labels and
        weights, and returns an object representing a (subset of a) minibatch on
        which one may calculate rates.

        The related `split_rate_context` method takes *two* `Tensor`s of
        predictions, labels and weights, the first for the "penalty" portion of
        the objective, and the second for the "constraint" portion. The purpose
        of splitting the context is to improve generalization performance: see
        [CotterEtAl18] for full details.

        The most important property of these objects is that they are
        subsettable: if you want to calculate a rate on e.g. only the
        negatively-labeled examples, or only those examples belonging to a
        certain protected class, then this can be accomplished via the `subset`
        method. However, you should *use great caution* with the `subset`
        method: if the desired subset is a very small proportion of the dataset
        (e.g. a protected class that's an extreme minority), then the resulting
        stochastic gradients will be noisy, and during training your model will
        converge very slowly. Instead, it is usually better (but less
        convenient) to create an entirely separate dataset for each rare subset
        (e.g. using the "filter" method of a `tf.train.Dataset`), and to
        construct each subset context directly from each such dataset.

    *   [binary_rates.py](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/rates/binary_rates.py):
        contains functions for constructing rates from contexts. These rates can
        then be combined into more complicated expressions using python
        arithmetic operators, or into constraints using comparison operators.

    *   [operations.py](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/rates/operations.py):
        contains functions for manipulating rate expressions, including
        `wrap_rate`, which can be used to convert a `Tensor` into a rate object,
        as well as `lower_bound` and `upper_bound`, which convert lists of rates
        into rates representing lower- and upper-bounds on all elements of the
        list.

    *   [loss.py](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/rates/loss.py):
        contains loss functions used in constructing rates. These can be passed
        as parameters to the optional `penalty_loss` and `constraint_loss`
        functions in "binary_rates.py" (above).

    *   [rate_minimization_problem.py](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/tensorflow_constrained_optimization/python/rates/rate_minimization_problem.py):
        contains the `RateMinimizationProblem` class, which constructs a
        `ConstrainedMinimizationProblem` (suitable for use by
        `ConstrainedOptimizer`s) from a rate expression to minimize, and a list
        of rate constraints to impose.

## Convex example using proxy constraints

This is a simple example of recall-constrained optimization on simulated data:
we seek a classifier that minimizes the average hinge loss while constraining
recall to be at least 90%.

We'll start with the required imports&mdash;notice the definition of `tfco`:

```python
import math
import numpy as np
from six.moves import xrange
import tensorflow as tf

import tensorflow_constrained_optimization as tfco
```

We'll next create a simple simulated dataset by sampling 1000 random
10-dimensional feature vectors from a Gaussian, finding their labels using a
random "ground truth" linear model, and then adding noise by randomly flipping
200 labels.

```python
# Create a simulated 10-dimensional training dataset consisting of 1000 labeled
# examples, of which 800 are labeled correctly and 200 are mislabeled.
num_examples = 1000
num_mislabeled_examples = 200
dimension = 10
# We will constrain the recall to be at least 90%.
recall_lower_bound = 0.9

# Create random "ground truth" parameters for a linear model.
ground_truth_weights = np.random.normal(size=dimension) / math.sqrt(dimension)
ground_truth_threshold = 0

# Generate a random set of features for each example.
features = np.random.normal(size=(num_examples, dimension)).astype(
    np.float32) / math.sqrt(dimension)
# Compute the labels from these features given the ground truth linear model.
labels = (np.matmul(features, ground_truth_weights) >
          ground_truth_threshold).astype(np.float32)
# Add noise by randomly flipping num_mislabeled_examples labels.
mislabeled_indices = np.random.choice(
    num_examples, num_mislabeled_examples, replace=False)
labels[mislabeled_indices] = 1 - labels[mislabeled_indices]
```

We're now ready to construct our model, and the corresponding optimization
problem. We'll use a linear model of the form **f(x) = w^T x - t**, where **w**
is the `weights`, and **t** is the `threshold`.

```python
# Create variables containing the model parameters.
weights = tf.Variable(tf.zeros(dimension), dtype=tf.float32, name="weights")
threshold = tf.Variable(0.0, dtype=tf.float32, name="threshold")

# Create the optimization problem.
constant_labels = tf.constant(labels, dtype=tf.float32)
constant_features = tf.constant(features, dtype=tf.float32)
predictions = tf.tensordot(constant_features, weights, axes=(1, 0)) - threshold
```

Now that we have the output of our linear model (in the `predictions` variable),
we can move on to constructing the optimization problem. At this point, there
are two ways to proceed:

1.  We can use the rate helpers provided by the TFCO library. This is the
    easiest way to construct optimization problems based on *rates* (where a
    "rate" is the proportion of training examples on which some event occurs).
2.  We could instead create an implementation of the
    `ConstrainedMinimizationProblem` interface. This is the most flexible
    approach. In particular, it is not limited to problems expressed in terms of
    rates.

We'll now consider each of these two options in turn.

### Option 1: rate helpers

The main motivation of TFCO is to make it easy to create and optimize
constrained problems written in terms of linear combinations of *rates*, where a
"rate" is the proportion of training examples on which an event occurs (e.g. the
false positive rate, which is the number of negatively-labeled examples on which
the model makes a positive prediction, divided by the number of
negatively-labeled examples). Our current example (minimizing a hinge relaxation
of the error rate subject to a recall constraint) is such a problem.

```python
context = tfco.rate_context(predictions, labels=constant_labels)
problem = tfco.RateMinimizationProblem(
    tfco.error_rate(context), [tfco.recall(context) >= recall_lower_bound])
```

Rate-construction functions (`error_rate` and `recall`, here) take two optional
named parameters&mdash;not used here&mdash;named `penalty_loss` and
`constraint_loss`, of which the former is used to define the proxy constraints,
and the latter the "true" constraints. These default to the hinge and zero-one
losses, respectively. The consequence of this is that we will attempt to
minimize the average hinge loss (a relaxation of the error rate using the
`penalty_loss`), while constraining the recall (using the `constraint_loss`) by
essentially learning how much we should penalize the hinge-constrained recall
(`penalty_loss`, again).

The `RateMinimizationProblem` class implements the
`ConstrainedMinimizationProblem` interface, and is constructed from a rate
expression to be minimized (the first parameter), subject to a list of rate
constraints (the second). Using this class is typically more convenient and
readable than constructing a `ConstrainedMinimizationProblem` manually: the
objects returned by `error_rate` and `recall`&mdash;and all other
rate-constructing and rate-combining functions&mdash;can be combined using
python arithmetic operators (e.g. "`0.5 * tfco.error_rate(context1) -
tfco.true_positive_rate(context2)`"), or converted into a constraint using a
comparison operator.

### Option 2: explicit ConstrainedMinimizationProblem

For problems that cannot be easily expressed using the rate helpers, one could
instead an explicit implementation of the `ConstrainedMinimizationProblem`
interface. The current task (minimizing the average hinge loss subject to a
recall constraint) is a rate-based problem, but for illustrative reasons we will
show how to create a `ConstrainedMinimizationProblem` for this task.

The constructor will takes three parameters: a `Tensor` containing the
classification labels (0 or 1) for every training example, another `Tensor`
containing the model's predictions on every training example (sometimes called
the "logits"), and the lower bound on recall that will be enforced using a
constraint.

As before, this implementation will contain both constraints *and* proxy
constraints: the former represents the constraint that the true recall (defined
in terms of the *number* of true positives) be at least `recall_lower_bound`,
while the latter represents the same constraint, but on a hinge approximation of
the recall.

```python
class ExampleProblem(tfco.ConstrainedMinimizationProblem):

  def __init__(self, labels, predictions, recall_lower_bound):
    self._labels = labels
    self._predictions = predictions
    self._recall_lower_bound = recall_lower_bound
    # The number of positively-labeled examples.
    self._positive_count = tf.reduce_sum(self._labels)

  @property
  def objective(self):
    return tf.losses.hinge_loss(labels=self._labels, logits=self._predictions)

  @property
  def constraints(self):
    # Recall that the labels are binary (0 or 1).
    true_positives = self._labels * tf.to_float(self._predictions > 0)
    true_positive_count = tf.reduce_sum(true_positives)
    recall = true_positive_count / self._positive_count
    # The constraint is (recall >= self._recall_lower_bound), which we convert
    # to (self._recall_lower_bound - recall <= 0) because
    # ConstrainedMinimizationProblems must always provide their constraints in
    # the form (tensor <= 0).
    #
    # The result of this function should be a tensor, with each element being
    # a quantity that is constrained to be nonpositive. We only have one
    # constraint, so we return a one-element tensor.
    return self._recall_lower_bound - recall

  @property
  def proxy_constraints(self):
    # Use 1 - hinge since we're SUBTRACTING recall in the constraint function,
    # and we want the proxy constraint function to be convex. Recall that the
    # labels are binary (0 or 1).
    true_positives = self._labels * tf.minimum(1.0, self._predictions)
    true_positive_count = tf.reduce_sum(true_positives)
    recall = true_positive_count / self._positive_count
    # Please see the corresponding comment in the constraints property.
    return self._recall_lower_bound - recall

problem = ExampleProblem(
    labels=constant_labels,
    predictions=predictions,
    recall_lower_bound=recall_lower_bound,
)
```

### Wrapping up

We're almost ready to train our model, but first we'll create a couple of
functions to measure its performance. We're interested in two quantities: the
average hinge loss (which we seek to minimize), and the recall (which we
constrain).

```python
def average_hinge_loss(labels, predictions):
  # Recall that the labels are binary (0 or 1).
  signed_labels = (labels * 2) - 1
  return np.mean(np.maximum(0.0, 1.0 - signed_labels * predictions))

def recall(labels, predictions):
  # Recall that the labels are binary (0 or 1).
  positive_count = np.sum(labels)
  true_positives = labels * (predictions > 0)
  true_positive_count = np.sum(true_positives)
  return true_positive_count / positive_count
```

As was mentioned earlier, the Lagrangian optimizer often suffices for problems
without proxy constraints, but proxy-Lagrangian optimizers are recommended for
problems *with* proxy constraints. Since this problem contains proxy
constraints, we use the `ProxyLagrangianOptimizer`.

For this problem, the constraint is fairly easy to satisfy, so we can use the
same "inner" optimizer (an `AdagradOptimizer` with a learning rate of 1) for
optimization of both the model parameters (`weights` and `threshold`), and the
internal parameters associated with the constraints (these are the analogues of
the Lagrange multipliers used by the `ProxyLagrangianOptimizer`). For more
difficult problems, it will often be necessary to use different optimizers, with
different learning rates (presumably found via a hyperparameter search): to
accomplish this, pass *both* the `optimizer` and `constraint_optimizer`
parameters to `ProxyLagrangianOptimizer`'s constructor.

Since this is a convex problem (both the objective and proxy constraint
functions are convex), we can just take the last iterate. Periodic snapshotting,
and the use of the `find_best_candidate_distribution` or
`find_best_candidate_index` functions, is generally only necessary for
non-convex problems (and even then, it isn't *always* necessary).

```python
with tf.Session() as session:
  optimizer = tfco.ProxyLagrangianOptimizer(
      optimizer=tf.train.AdagradOptimizer(learning_rate=1.0))
  train_op = optimizer.minimize(problem)

  session.run(tf.global_variables_initializer())
  for ii in xrange(1000):
    session.run(train_op)

  trained_weights, trained_threshold = session.run((weights, threshold))

trained_predictions = np.matmul(features, trained_weights) - trained_threshold
print("Constrained average hinge loss = %f" % average_hinge_loss(
    labels, trained_predictions))
print("Constrained recall = %f" % recall(labels, trained_predictions))
```

Running the above code gives the following output (due to the randomness of the
dataset, you'll get a different result when you run it):

```none
Constrained average hinge loss = 0.683846
Constrained recall = 0.899791
```

As we hoped, the recall is extremely close to 90%&mdash;and, thanks to the fact
that the optimizer uses a (hinge) proxy constraint only when needed, and the
actual (zero-one) constraint whenever possible, this is the *true* recall, not a
hinge approximation.

For comparison, let's try optimizing the same problem *without* the recall
constraint:

```python
with tf.Session() as session:
  optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
  # For optimizing the unconstrained problem, we just minimize the "objective"
  # portion of the minimization problem. We could instead have used the
  # "minimize_unconstrained" method of a ConstrainedOptimizer.
  train_op = optimizer.minimize(problem.objective)

  session.run(tf.global_variables_initializer())
  for ii in xrange(1000):
    session.run(train_op)

  trained_weights, trained_threshold = session.run((weights, threshold))

trained_predictions = np.matmul(features, trained_weights) - trained_threshold
print("Unconstrained average hinge loss = %f" % average_hinge_loss(
    labels, trained_predictions))
print("Unconstrained recall = %f" % recall(labels, trained_predictions))
```

This code gives the following output (again, you'll get a different answer,
since the dataset is random):

```none
Unconstrained average hinge loss = 0.612755
Unconstrained recall = 0.801670
```

Because there is no constraint, the unconstrained problem does a better job of
minimizing the average hinge loss, but naturally doesn't approach 90% recall.

## More examples

*   [Recall_constraint.ipynb](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/examples/jupyter_notebooks/Recall_constraint.ipynb):
    **Start here!** This is the above simple example as a
    [Jupyter](https://jupyter.org/) notebook.
*   [Fairness_adult.ipynb](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/examples/jupyter_notebooks/Fairness_adult.ipynb):
    This notebook shows how to train classifiers for fairness constraints on the UCI Adult dataset using rate-constraints.
*   [Generalization_communities.ipynb](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/examples/jupyter_notebooks/Generalization_communities.ipynb):
    This notebook shows how to improve fairness generalization performance on the UCI Communities and Crime dataset by using the split dataset approach.
*   [Minibatch_training.ipynb](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/examples/jupyter_notebooks/Minibatch_training.ipynb):
    This notebook describes how to solve a rate-constrained training problem using *minibatches*. The notebook focuses on problems where one wishes to impose constraints on small groups of examples, and shows how one can speed up the convergence of the optimizer by using separate streams of minibatches for each group.
*   [Oscillation_compas.ipynb](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/examples/jupyter_notebooks/Oscillation_compas.ipynb):
    This notebook shows instances where it is possible that the iterates don't converge when using the Lagrangian approach to training with fairness constraintsthus motivating more careful selection of solutions or the use of stochastic classifiers.
*   [Post_processing.ipynb](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/examples/jupyter_notebooks/Post_processing.ipynb):
    This notebook describes how to use the shrinking procedure of [CoJiSr19] to post-process the iterates of a constrained optimizer and construct a stochastic classifier from them. For applications where a stochastic classifier is not acceptable, we show how to use a heuristic provided by [CoJiSr19] to pick the best deterministic classifier from the iterates of the optimizer.
*   [Churn.ipynb](https://github.com/google-research/tensorflow_constrained_optimization/tree/master/examples/jupyter_notebooks/Churn.ipynb):
    This notebook describes how to use rate constraints to enforce low-churn classification. That is, to train for accuracy while ensuring the predictions don't differ by much compared to a baseline model.


