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
"""A library for performing constrained optimization in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_constrained_optimization.python.candidates import find_best_candidate_distribution
from tensorflow_constrained_optimization.python.candidates import find_best_candidate_index
from tensorflow_constrained_optimization.python.constrained_minimization_problem import ConstrainedMinimizationProblem
from tensorflow_constrained_optimization.python.rates.binary_rates import inverse_precision_at_recall
from tensorflow_constrained_optimization.python.rates.binary_rates import pr_auc
from tensorflow_constrained_optimization.python.rates.binary_rates import precision_at_recall
from tensorflow_constrained_optimization.python.rates.binary_rates import recall_at_precision
from tensorflow_constrained_optimization.python.rates.binary_rates import roc_auc
from tensorflow_constrained_optimization.python.rates.general_rates import accuracy_rate
from tensorflow_constrained_optimization.python.rates.general_rates import error_rate
from tensorflow_constrained_optimization.python.rates.general_rates import f_score
from tensorflow_constrained_optimization.python.rates.general_rates import f_score_ratio
from tensorflow_constrained_optimization.python.rates.general_rates import false_negative_proportion
from tensorflow_constrained_optimization.python.rates.general_rates import false_negative_rate
from tensorflow_constrained_optimization.python.rates.general_rates import false_positive_proportion
from tensorflow_constrained_optimization.python.rates.general_rates import false_positive_rate
from tensorflow_constrained_optimization.python.rates.general_rates import negative_prediction_rate
from tensorflow_constrained_optimization.python.rates.general_rates import positive_prediction_rate
from tensorflow_constrained_optimization.python.rates.general_rates import precision
from tensorflow_constrained_optimization.python.rates.general_rates import precision_ratio
from tensorflow_constrained_optimization.python.rates.general_rates import true_negative_proportion
from tensorflow_constrained_optimization.python.rates.general_rates import true_negative_rate
from tensorflow_constrained_optimization.python.rates.general_rates import true_positive_proportion
from tensorflow_constrained_optimization.python.rates.general_rates import true_positive_rate
from tensorflow_constrained_optimization.python.rates.keras import KerasLayer
from tensorflow_constrained_optimization.python.rates.keras import KerasMetricWrapper
from tensorflow_constrained_optimization.python.rates.keras import KerasPlaceholder
from tensorflow_constrained_optimization.python.rates.loss import BinaryClassificationLoss
from tensorflow_constrained_optimization.python.rates.loss import HingeLoss
from tensorflow_constrained_optimization.python.rates.loss import Loss
from tensorflow_constrained_optimization.python.rates.loss import MulticlassLoss
from tensorflow_constrained_optimization.python.rates.loss import SoftmaxCrossEntropyLoss
from tensorflow_constrained_optimization.python.rates.loss import SoftmaxLoss
from tensorflow_constrained_optimization.python.rates.loss import ZeroOneLoss
from tensorflow_constrained_optimization.python.rates.operations import lower_bound
from tensorflow_constrained_optimization.python.rates.operations import upper_bound
from tensorflow_constrained_optimization.python.rates.operations import wrap_rate
from tensorflow_constrained_optimization.python.rates.rate_minimization_problem import RateMinimizationProblem
from tensorflow_constrained_optimization.python.rates.subsettable_context import multiclass_rate_context
from tensorflow_constrained_optimization.python.rates.subsettable_context import multiclass_split_rate_context
from tensorflow_constrained_optimization.python.rates.subsettable_context import rate_context
from tensorflow_constrained_optimization.python.rates.subsettable_context import split_rate_context
from tensorflow_constrained_optimization.python.train.constrained_optimizer import ConstrainedOptimizerV1
from tensorflow_constrained_optimization.python.train.constrained_optimizer import ConstrainedOptimizerV2
from tensorflow_constrained_optimization.python.train.lagrangian_optimizer import create_lagrangian_loss
from tensorflow_constrained_optimization.python.train.lagrangian_optimizer import LagrangianOptimizerV1
from tensorflow_constrained_optimization.python.train.lagrangian_optimizer import LagrangianOptimizerV2
from tensorflow_constrained_optimization.python.train.proxy_lagrangian_optimizer import create_proxy_lagrangian_loss
from tensorflow_constrained_optimization.python.train.proxy_lagrangian_optimizer import ProxyLagrangianOptimizerV1
from tensorflow_constrained_optimization.python.train.proxy_lagrangian_optimizer import ProxyLagrangianOptimizerV2

# The "true positive rate" is the same thing as the "recall", so we allow it to
# be accessed by either name.
recall = true_positive_rate

# By default, we use V2 optimizers. These aliases are purely for convenience: in
# general, you should prefer to explicitly specify either a V1 or a V2 optimizer
# (in case there's ever a V3, in which case we'll update these aliases).
ConstrainedOptimizer = ConstrainedOptimizerV2
LagrangianOptimizer = LagrangianOptimizerV2
ProxyLagrangianOptimizer = ProxyLagrangianOptimizerV2
