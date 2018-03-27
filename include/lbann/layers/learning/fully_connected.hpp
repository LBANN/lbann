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

#ifndef LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
#define LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED

#include "lbann/layers/learning/learning.hpp"
#include "lbann/layers/activations/activation.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/models/model.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/fan_in_fan_out_initializers.hpp"
#include "lbann/utils/cublas_wrapper.hpp"
#include <string>
#include <sstream>

namespace lbann {

enum class device {CPU, CUDA};

/** Fully-connected layer.
 *  This layer applies an affine transformation.
 */
template <data_layout T_layout>
class fully_connected_layer : public learning_layer {
 private:

  /** Scaling factor for bias term.
   *  If the scaling factor is zero, bias is not applied.
   */
  DataType m_bias_scaling_factor;

  /** Linearity gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the linearity weights (i.e. its matrix weights).
   */
  AbsDistMat* m_linearity_gradient;
  /** Bias weights gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the bias weights.
   */
  AbsDistMat* m_bias_gradient;

#ifdef LBANN_HAS_CUDNN
  /** GPU memory for linearity gradient. */
  cudnn::matrix m_linearity_gradient_d;
  /** GPU memory for bias gradient. */
  cudnn::matrix m_bias_gradient_d;
#endif // __LIB_CUNN

 public:

  fully_connected_layer(lbann_comm *comm,
                        int num_neurons,  // TODO: accept a vector for neuron dims
                        weights* weight = nullptr,
                        bool has_bias = true,
                        cudnn::cudnn_manager *cudnn = nullptr)
    : learning_layer(comm),
      m_linearity_gradient(nullptr),
      m_bias_gradient(nullptr) {

    // Initialize neuron tensor dimensions
    this->m_num_neurons = num_neurons;
    this->m_num_neuron_dims = 1;
    this->m_neuron_dims.assign(1, this->m_num_neurons);

    // Initialize bias
    m_bias_scaling_factor = has_bias ? DataType(1) : DataType(0);

#ifdef LBANN_HAS_CUDNN
    if (cudnn && T_layout == data_layout::DATA_PARALLEL) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
#endif // LBANN_HAS_CUDNN
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} +
     " fully_connected; num_neurons: "
     + std::to_string(this->m_num_neurons)
     + " has_bias: " + std::to_string(this->m_bias_scaling_factor)
     + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  fully_connected_layer(const fully_connected_layer& other) :
    learning_layer(other),
    m_bias_scaling_factor(other.m_bias_scaling_factor) {
    
    // Deep matrix copies
    m_linearity_gradient = other.m_linearity_gradient;
    m_bias_gradient = other.m_bias_gradient;
    if (m_linearity_gradient != nullptr) {
      m_linearity_gradient = m_linearity_gradient->Copy();
    }
    if (m_bias_gradient != nullptr) {
      m_bias_gradient = m_bias_gradient->Copy();
    }

#ifdef LBANN_HAS_CUDNN
    m_linearity_gradient_d = other.m_linearity_gradient_d;
    m_bias_gradient_d = other.m_bias_gradient_d;
#endif // LBANN_HAS_CUDNN

  }

  fully_connected_layer& operator=(const fully_connected_layer& other) {
    learning_layer::operator=(other);
    m_bias_scaling_factor = other.m_bias_scaling_factor;

    // Deep matrix copies
    deallocate_matrices();
    m_linearity_gradient = other.m_linearity_gradient;
    m_bias_gradient = other.m_bias_gradient;
    if (m_linearity_gradient != nullptr) {
      m_linearity_gradient = m_linearity_gradient->Copy();
    }
    if (m_bias_gradient != nullptr) {
      m_bias_gradient = m_bias_gradient->Copy();
    }

  #ifdef LBANN_HAS_CUDNN
    m_linearity_gradient_d = other.m_linearity_gradient_d;
    m_bias_gradient_d = other.m_bias_gradient_d;
  #endif // LBANN_HAS_CUDNN

    return *this;
  }

  ~fully_connected_layer() override {
    deallocate_matrices();
  }

  fully_connected_layer* copy() const override {
    return new fully_connected_layer(*this);
  }

  std::string get_type() const override { return "fully connected"; }

  data_layout get_data_layout() const override { return T_layout; }

  void setup_matrices(const El::Grid& grid) override;

  void setup_dims() override {
    // Store neuron tensor dimensions
    const int num_neurons = this->m_num_neurons;
    const int num_neuron_dims = this->m_num_neuron_dims;
    const std::vector<int> neuron_dims = this->m_neuron_dims;

    // Initialize previous neuron tensor dimensions
    learning_layer::setup_dims();

    // Initialize neuron tensor dimensions
    this->m_num_neurons = num_neurons;
    this->m_num_neuron_dims = num_neuron_dims;
    this->m_neuron_dims = neuron_dims;
  }

  void setup_data() override {
    learning_layer::setup_data();

    // Initialize default weights if none are provided
    if (this->m_weights.size() > 2) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to setup " << m_name << " with an invalid number of weights";
      throw lbann_exception(err.str());
    }
    this->m_weights.resize(2, nullptr);
    if (this->m_weights[0] == nullptr) {
      this->m_weights[0] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[0]->set_name(this->m_name + "_linearity_weights");
      this->m_weights[0]->set_initializer(new he_normal_initializer(this->m_comm));
      this->m_weights[0]->set_optimizer(m_model->create_optimizer());
      this->m_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights[1] == nullptr) {
      this->m_weights[1] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[1]->set_name(this->m_name + "_bias_weights");
      this->m_weights[1]->set_optimizer(m_model->create_optimizer());
      this->m_model->add_weights(this->m_weights[1]);
    }

    // Initialize Glorot or He weight initialization
    auto* cast_initializer
      = dynamic_cast<fan_in_fan_out_initializer*>(&this->m_weights[0]->get_initializer());
    if (cast_initializer != nullptr) {
      cast_initializer->set_fan_in(this->m_num_prev_neurons);
      cast_initializer->set_fan_out(this->m_num_neurons);
    }

    // Setup weights
    // Note: linearity matrix is duplicated across processes unless
    // the data layout is model-parallel.
    El::Distribution col_dist = El::STAR;
    El::Distribution row_dist = El::STAR;
    if (get_data_layout() == data_layout::MODEL_PARALLEL) {
      col_dist = El::MC;
      row_dist = El::MR;
    }
    this->m_weights[0]->setup(this->m_num_neurons,
                              this->m_num_prev_neurons,
                              col_dist, row_dist);
    this->m_weights[1]->setup(this->m_num_neurons,
                              1,
                              get_activations().DistData().colDist,
                              El::STAR);

    // Setup weight gradients
    El::Zeros(*this->m_linearity_gradient,
              this->m_weights[0]->get_matrix_height(),
              this->m_weights[0]->get_matrix_width());
    El::Zeros(*this->m_bias_gradient,
              this->m_weights[1]->get_matrix_height(),
              this->m_weights[1]->get_matrix_width());

  }

  void setup_gpu() override {
    learning_layer::setup_gpu();
#ifndef LBANN_HAS_CUDNN
    throw lbann_exception("fully_connected_layer: CUDA not detected");
#else
    m_linearity_gradient_d = cudnn::matrix(m_cudnn,
                                           m_linearity_gradient->Height(),
                                           m_linearity_gradient->Width());
    if(m_bias_scaling_factor != DataType(0)) {
      m_bias_gradient_d = cudnn::matrix(m_cudnn,
                                        m_bias_gradient->Height(),
                                        m_bias_gradient->Width());
    }
#endif // LBANN_HAS_CUDNN
  }

  void fp_compute() override {
    if(this->m_using_gpus) {
      fp_compute_cuda();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
      bp_compute_cuda();
    } else {
      bp_compute_cpu();
    }
  }

 private:

  /** CPU implementation of forward prop computation. */
  void fp_compute_cpu();
  /** CPU implementation of backward prop computation. */
  void bp_compute_cpu();

  /** GPU implementation of forward prop computation. */
  void fp_compute_cuda() {
#ifndef LBANN_HAS_CUDNN
    throw lbann_exception("fully_connected: CUDA not detected");
#else

    // GPU matrices
    const auto& linearity_d = m_weights[0]->get_values_gpu();
    const auto& input_d = this->m_prev_activations_d[0];
    auto& output_d = this->m_activations_d[0];
    
    // Matrix parameters
    const int input_size = get_num_prev_neurons();
    const int output_size = get_num_neurons();
    const int mini_batch_size = m_mini_batch_size_per_gpu;
    const int num_gpus = this->m_cudnn->get_num_gpus();
    const int input_ldim = input_d.get_leading_dim();
    const int output_ldim = output_d.get_leading_dim();

    // Stop early if possible
    if (mini_batch_size == 0) { return; }

    // Apply linearity
    for (int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   output_size, mini_batch_size, input_size,
                   DataType(1),
                   linearity_d[i], output_size,
                   input_d.get_locked_data(i), input_ldim,
                   DataType(0),
                   output_d.get_data(i), output_ldim);
    }

    // Apply bias if needed
    if(m_bias_scaling_factor != DataType(0)) {
      const auto& bias_d = m_weights[1]->get_values_gpu();
      
      // Initialize work space with ones
      cudnn::matrix ones_d(this->m_cudnn);
      ones_d.attach_to_work_spaces(mini_batch_size);
      m_cudnn->set_on_gpus(ones_d.get_data(), DataType(1), mini_batch_size);

      // Apply bias with outer product
      for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                     CUBLAS_OP_N, CUBLAS_OP_T,
                     output_size, mini_batch_size, 1,
                     DataType(1),
                     bias_d[i], output_size,
                     ones_d.get_data(i), mini_batch_size,
                     DataType(1),
                     output_d.get_data(i), output_ldim);
      }

    }
#endif // LBANN_HAS_CUDNN
  }

  /** GPU implementation of backward prop computation. */
  void bp_compute_cuda() {
#ifndef LBANN_HAS_CUDNN
    throw lbann_exception("fully_connected: CUDA not detected");
#else

    // GPU matrices
    const auto& linearity_d = m_weights[0]->get_values_gpu();
    const auto& input_d = this->m_prev_activations_d[0];
    const auto& gradient_wrt_output_d = this->m_prev_error_signals_d[0];
    auto& gradient_wrt_input_d = this->m_error_signals_d[0];
    
    // Matrix parameters
    const int input_size = get_num_prev_neurons();
    const int output_size = get_num_neurons();
    const int mini_batch_size = m_mini_batch_size_per_gpu;
    const int num_gpus = this->m_cudnn->get_num_gpus();
    const int input_ldim = input_d.get_leading_dim();
    const int gradient_wrt_output_ldim = gradient_wrt_output_d.get_leading_dim();
    const int gradient_wrt_input_ldim = gradient_wrt_input_d.get_leading_dim();

    // Compute gradient w.r.t. bias if needed
    optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
    if (m_bias_scaling_factor != DataType(0)
        && bias_optimizer != nullptr) {

      // Initialize work space with ones
      cudnn::matrix ones_d(this->m_cudnn);
      ones_d.attach_to_work_spaces(mini_batch_size);
      m_cudnn->set_on_gpus(ones_d.get_data(), DataType(1), mini_batch_size);

      // Obtain gradient with a sum over rows
      for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        cublas::gemv(this->m_cudnn->get_cublas_handle(i),
                     CUBLAS_OP_N, 
                     output_size, mini_batch_size,
                     DataType(1),
                     gradient_wrt_output_d.get_locked_data(i), gradient_wrt_output_ldim,
                     ones_d.get_data(i), 1,
                     DataType(0),
                     m_bias_gradient_d.get_data(i), 1);
      }
      bias_optimizer->add_to_gradient_staging(
        m_bias_gradient_d,
        m_bias_scaling_factor / this->m_model->get_effective_mini_batch_size());
    }
      
    // Compute gradient w.r.t. linearity if needed
    optimizer* linearity_optimizer = this->m_weights[0]->get_optimizer();
    if (linearity_optimizer != nullptr) {
      for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                     CUBLAS_OP_N, CUBLAS_OP_T,
                     output_size, input_size, mini_batch_size,
                     DataType(1),
                     gradient_wrt_output_d.get_locked_data(i), gradient_wrt_output_ldim,
                     input_d.get_locked_data(i), input_ldim,
                     DataType(0),
                     m_linearity_gradient_d.get_data(i), output_size);
      }
      linearity_optimizer->add_to_gradient_staging(
        m_linearity_gradient_d,
        DataType(1) / this->m_model->get_effective_mini_batch_size());
    }

    // Compute gradient w.r.t. input
    if (mini_batch_size != 0) {
      for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     input_size, mini_batch_size, output_size,
                     DataType(1),
                     linearity_d[i], output_size,
                     gradient_wrt_output_d.get_locked_data(i), gradient_wrt_output_ldim,
                     DataType(1),
                     gradient_wrt_input_d.get_data(i), gradient_wrt_input_ldim);
      }
    }

#endif // LBANN_HAS_CUDNN
  }

  /** Deallocate distributed matrices. */
  void deallocate_matrices() {
    if (m_linearity_gradient != nullptr) delete m_linearity_gradient;
    if (m_bias_gradient != nullptr) delete m_bias_gradient;
  }

};

} // namespace lbann

#endif // LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
