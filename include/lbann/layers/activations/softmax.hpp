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

#ifndef LBANN_LAYER_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYER_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/models/model.hpp"
#include <unistd.h>
#include <string>

#include <cassert>

// Output has minimum value to avoid denormalized floats
#define LBANN_ENABLE_SOFTMAX_CUTOFF

namespace lbann {

#ifdef LBANN_HAS_CUDNN
namespace softmax_cuda {
/** Apply minimum cutoff to activation entries.
 *  A minimum output value helps avoid denormalized floats.
 */
void fp_cutoff(cudnn::cudnn_manager& cudnn,
               DataType* activations,
               El::Int h, El::Int w,
               DataType min_output);
/** Error signal correction if activations have minimum cutoff. */
void bp_cutoff(cudnn::cudnn_manager& cudnn,
               const DataType* activations,
               DataType* error_signals,
               El::Int h, El::Int w,
               DataType min_output);
} // namespace softmax
#endif // LBANN_HAS_CUDNN

/** Softmax layer. */
template <data_layout T_layout, El::Device Dev>
class softmax_layer : public activation_layer {

 private:

  /** Workspace for column-wise reductions. */
  AbsDistMat *m_workspace;

  /** Lower bound for outputs.
   *  This should be sufficiently large to avoid denormalized
   *  floats.
   */
  DataType m_min_output;

 public:

  softmax_layer(lbann_comm *comm,
                cudnn::cudnn_manager *cudnn=nullptr)
    : activation_layer(comm),
      m_workspace(nullptr),
      m_min_output(std::sqrt(std::numeric_limits<DataType>::min())) {
    this->m_cudnn = cudnn;
  }

  softmax_layer(const softmax_layer& other)
    : activation_layer(other),
      m_min_output(other.m_min_output) {

    // Matrix deep copy
    m_workspace = other.m_workspace;
    if (m_workspace != nullptr) { m_workspace = m_workspace->Copy(); }

    // Copy GPU objects
    this->m_cudnn = other.m_cudnn;

  }

  softmax_layer& operator=(const softmax_layer& other) {
    activation_layer::operator=(other);
    m_min_output = other.m_min_output;

    // Deep matrix copy
    if (m_workspace != nullptr) { delete m_workspace; }
    m_workspace = other.m_workspace;
    if (m_workspace != nullptr) { m_workspace = m_workspace->Copy(); }

    // Copy GPU objects
    this->m_cudnn = other.m_cudnn;
    this->m_using_gpus = other.m_using_gpus;

  }

  ~softmax_layer() override {
    if (m_workspace != nullptr) { delete m_workspace; }
  }

  softmax_layer* copy() const override { return new softmax_layer(*this); }
  std::string get_type() const override { return "softmax"; }

  std::string get_description() const override {
    return std::string {} + " softmax" + " dataLayout: "
           + this->get_data_layout_string(get_data_layout());
  }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  void setup_matrices(const El::Grid& grid) override;

  void setup_data() override {
    activation_layer::setup_data();
    const int mini_batch_size = this->m_model->get_max_mini_batch_size();
    m_workspace->Resize(1, mini_batch_size);
  }

  void fp_setup_data(int mini_batch_size) override {
    activation_layer::fp_setup_data(mini_batch_size);
    m_workspace->Resize(1, mini_batch_size);
  }

  void fp_compute() override;
  void bp_compute() override;

  virtual void fp_compute_cpu() {

    // Local matrices
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();
    auto& local_workspace = m_workspace->Matrix();

    // Matrix parameters
    const El::Int local_height = local_input.Height();
    const El::Int local_width = local_input.Width();

    // Find maximum entry in each column
    #pragma omp parallel for
    for(El::Int col = 0; col < local_width; ++col) {
      DataType max_entry = local_input(0, col);
      for(El::Int row = 1; row < local_height; ++row) {
        max_entry = std::max(max_entry, local_input(row, col));
      }
      local_workspace(0, col) = max_entry;
    }
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm(),
                      El::mpi::MAX);

    // Exponentiate activations and compute column sums
    // Note: Subtracting by the column max prevents activations from
    // blowing up. Large negative values underflow to 0.
    #pragma omp parallel for
    for (El::Int col = 0; col < local_width; ++col) {
      const DataType shift = local_workspace(0, col);
      DataType sum = 0;
      for (El::Int row = 0; row < local_height; ++row) {
        const DataType x = local_input(row, col);
        const DataType y = std::exp(x - shift);
        local_output(row, col) = y;
        sum += y;
      }
      local_workspace(0, col) = sum;
    }
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm());

    // Divide activations by column sums
    // Note: Small values are rounded to minimum output value to avoid
    // denormalized floats.
    #pragma omp parallel for
    for (El::Int col = 0; col < local_width; ++col) {
      const DataType scale = DataType(1) / local_workspace(0, col);
      for (El::Int row = 0; row < local_height; ++row) {
        DataType& y = local_output(row, col);
        y *= scale;
      #ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
        y = std::max(y, m_min_output);
      #endif // LBANN_ENABLE_SOFTMAX_CUTOFF
      }
    }

  }

  virtual void bp_compute_cpu() {

    // Local matrices
    const DMat<Dev>& local_output = get_local_activations();
    const DMat<Dev>& local_gradient_wrt_output = get_local_prev_error_signals();
    DMat<Dev>& local_gradient_wrt_input = get_local_error_signals();
    DMat<Dev>& local_workspace = m_workspace->Matrix();

    // Matrix parameters
    const El::Int local_height = local_output.Height();
    const El::Int local_width = local_output.Width();

    // Compute dot products between output and gradient w.r.t. output
    for (El::Int col = 0; col < local_width; ++col) {
      const auto& y = local_output(El::ALL, El::IR(col));
      const auto& dy = local_gradient_wrt_output(El::ALL, El::IR(col));
      local_workspace(0, col) = El::Dot(y, dy);
    }
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm());

    // Compute gradient w.r.t. input
    #pragma omp parallel for
    for (El::Int col = 0; col < local_width; ++col) {
      const DataType y_dot_dy = local_workspace(0, col);
      for (El::Int row = 0; row < local_height; ++row) {
        const DataType y = local_output(row, col);
        const DataType dy = local_gradient_wrt_output(row, col);
        DataType dx = y * (dy - y_dot_dy);
      #ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
        if (y <= m_min_output) { dx = DataType(0); }
      #endif
        local_gradient_wrt_input(row, col) += dx;
      }
    }

  }

};

} // namespace lbann

#endif // LBANN_LAYER_SOFTMAX_HPP_INCLUDED
