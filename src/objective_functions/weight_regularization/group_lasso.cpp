////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/objective_functions/weight_regularization/group_lasso.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

void group_lasso_weight_regularization::setup(model& m) {
  objective_function_term::setup(m);

  // Check that term has no layer pointers
  if (!m_layers.empty()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to setup group lasso weight regularization with layer pointers";
    throw lbann_exception(err.str());
  }

  // Add all weights in model if no weights pointers are provided
  if (m_weights.empty()) {
    for (weights* w : m.get_weights()) {
      if (w->get_optimizer() != nullptr) {
        m_weights.push_back(w);
      }
    }
  }

}

void group_lasso_weight_regularization::start_evaluation() {

}

EvalType group_lasso_weight_regularization::finish_evaluation() {
  if (m_scale_factor == EvalType(0)) { return EvalType(0); }
  EvalType value = EvalType(0);
  CPUMat sqsums;
  for (weights* w : m_weights) {

    // Get matrices
    const AbsDistMat& values = w->get_values();
    const Mat& values_local = values.LockedMatrix();
    const int local_height = values_local.Height();
    const int local_width = values_local.Width();

    // Compute sum of squares of each column
    sqsums.Resize(1, local_width);
    #pragma omp parallel for
    for (int col = 0; col < local_width; ++col) {
      DataType sqsum = EvalType(0);
      for (int row = 0; row < local_height; ++row) {
        const DataType val = values_local(row, col);
        sqsum += val * val;
      }
      sqsums(0, col) = sqsum;
    }
    get_comm().allreduce(sqsums.Buffer(), local_width, sqsums.Buffer(),
                          values.ColComm());

    // Compute group lasso term
    EvalType w_sum = EvalType(0);
    for (int col = 0; col < local_width; ++col) {
      w_sum += std::sqrt(sqsums(0, col));
    }
    value += get_comm().allreduce(w_sum, values.RowComm());

  }
  return m_scale_factor * value;
}

void group_lasso_weight_regularization::compute_weight_regularization() {
  if (m_scale_factor == EvalType(0)) { return; }
  CPUMat sqsums;
  AbsDistMat* gradient;
  for (weights* w : m_weights) {

    // Get matrices
    const AbsDistMat& values = w->get_values();
    const Mat& values_local = values.LockedMatrix();
    const int local_height = values_local.Height();
    const int local_width = values_local.Width();

    // Compute sum of squares of each column
    sqsums.Resize(1, local_width);
    #pragma omp parallel for
    for (int col = 0; col < local_width; ++col) {
      DataType sqsum = DataType(0);
      for (int row = 0; row < local_height; ++row) {
        const DataType val = values_local(row, col);
        sqsum += val * val;
      }
      sqsums(0, col) = sqsum;
    }
    get_comm().allreduce(sqsums.Buffer(), local_width, sqsums.Buffer(),
                         values.ColComm());

    // Compute gradient
    gradient = values.Copy();
    Mat& gradient_local = gradient->Matrix();
    for (int col = 0; col < local_width; ++col) {
      if (sqsums(0, col) != DataType(0)) {
        sqsums(0, col) = 1 / std::sqrt(sqsums(0, col));
      } else {
        sqsums(0, col) = DataType(0);
      }
    }
    #pragma omp parallel for collapse(2)
    for (int col = 0; col < local_width; ++col) {
      for (int row = 0; row < local_height; ++row) {
        gradient_local(row, col) = values_local(row, col) * sqsums(0, col);
      }
    }
    w->get_optimizer()->add_to_gradient(*gradient, m_scale_factor);
    delete gradient;

  }
}

}  // namespace lbann
