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
from tensorflow_constrained_optimization.python.rates.binary_rates import accuracy_rate
from tensorflow_constrained_optimization.python.rates.binary_rates import error_rate
from tensorflow_constrained_optimization.python.rates.binary_rates import f_score_ratio
from tensorflow_constrained_optimization.python.rates.binary_rates import false_negative_rate
from tensorflow_constrained_optimization.python.rates.binary_rates import false_positive_rate
from tensorflow_constrained_optimization.python.rates.binary_rates import negative_prediction_rate
from tensorflow_constrained_optimization.python.rates.binary_rates import positive_prediction_rate
from tensorflow_constrained_optimization.python.rates.binary_rates import precision_ratio
from tensorflow_constrained_optimization.python.rates.binary_rates import recall
from tensorflow_constrained_optimization.python.rates.binary_rates import roc_auc_lower_bound
from tensorflow_constrained_optimization.python.rates.binary_rates import roc_auc_upper_bound
from tensorflow_constrained_optimization.python.rates.binary_rates import true_negative_rate
from tensorflow_constrained_optimization.python.rates.binary_rates import true_positive_rate
from tensorflow_constrained_optimization.python.rates.loss import HingeLoss
from tensorflow_constrained_optimization.python.rates.loss import ZeroOneLoss
from tensorflow_constrained_optimization.python.rates.operations import lower_bound
from tensorflow_constrained_optimization.python.rates.operations import upper_bound
from tensorflow_constrained_optimization.python.rates.operations import wrap_rate
from tensorflow_constrained_optimization.python.rates.rate_minimization_problem import RateMinimizationProblem
from tensorflow_constrained_optimization.python.rates.subsettable_context import rate_context
from tensorflow_constrained_optimization.python.rates.subsettable_context import split_rate_context
from tensorflow_constrained_optimization.python.train.constrained_optimizer import ConstrainedOptimizer
from tensorflow_constrained_optimization.python.train.lagrangian_optimizer import create_lagrangian_loss
from tensorflow_constrained_optimization.python.train.lagrangian_optimizer import LagrangianOptimizer
from tensorflow_constrained_optimization.python.train.proxy_lagrangian_optimizer import ProxyLagrangianOptimizer
