////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYERS_TRANSFORM_CONCATENATE_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_CONCATENATE_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Concatenate tensors along specified dimension. */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class concatenate_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  concatenate_layer(lbann_comm *comm, El::Int concat_dim);
  concatenate_layer(const concatenate_layer& other);
  concatenate_layer& operator=(const concatenate_layer& other);

  concatenate_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

  description get_description() const override;

protected:

  void setup_pointers() override;
  void setup_matrices(const El::Grid& grid) override;
  void setup_dims() override;

  void fp_setup_outputs(El::Int mini_batch_size) override;
  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override;
  void fp_compute() override;
  void bp_compute() override;

private:

  /** Tensor dimension to concatenate. */
  El::Int m_concat_dim;
  /** Concatenate points for each child layer. */
  std::vector<El::Int> m_concat_points;

  /** View into input tensor. */
  std::unique_ptr<AbsDistMatrixType> m_input_v;
  /** View into output tensor. */
  std::unique_ptr<AbsDistMatrixType> m_output_v;

};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
concatenate_layer<TensorDataType,Layout,Device>::concatenate_layer(
  lbann_comm *comm,
  El::Int concat_dim)
  : data_type_layer<TensorDataType>(comm),
    m_concat_dim(concat_dim) {
  this->m_expected_num_parent_layers = -1; // No limit on parents
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
concatenate_layer<TensorDataType,Layout,Device>::concatenate_layer(
  const concatenate_layer& other)
  : data_type_layer<TensorDataType>(other),
    m_concat_dim(other.m_concat_dim),
    m_concat_points(other.m_concat_points) {
  m_input_v.reset(other.m_input_v ? other.m_input_v->Copy() : nullptr);
  m_output_v.reset(other.m_output_v ? other.m_output_v->Copy() : nullptr);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
concatenate_layer<TensorDataType,Layout,Device>& concatenate_layer<TensorDataType,Layout,Device>::operator=(
  const concatenate_layer& other) {
  data_type_layer<TensorDataType>::operator=(other);
  m_concat_dim = other.m_concat_dim;
  m_concat_points = other.m_concat_points;
  m_input_v.reset(other.m_input_v ? other.m_input_v->Copy() : nullptr);
  m_output_v.reset(other.m_output_v ? other.m_output_v->Copy() : nullptr);
  return *this;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
concatenate_layer<TensorDataType, Layout,Device>* concatenate_layer<TensorDataType,Layout,Device>::copy() const {
  return new concatenate_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string concatenate_layer<TensorDataType,Layout,Device>::get_type() const {
  return "concatenate";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout concatenate_layer<TensorDataType,Layout,Device>::get_data_layout() const {
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device concatenate_layer<TensorDataType,Layout,Device>::get_device_allocation() const {
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description concatenate_layer<TensorDataType,Layout,Device>::get_description() const {
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Concatenate dimension", m_concat_dim);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::setup_pointers() {
  data_type_layer<TensorDataType>::setup_pointers();
  if (this->get_num_parents() < 1) {
    LBANN_ERROR(get_type()," layer \"",this->get_name(),"\" ",
                "has no parents");
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::setup_matrices(const El::Grid& grid) {
  data_type_layer<TensorDataType>::setup_matrices(grid);
  const auto& input = this->get_prev_activations();
  m_input_v.reset(input.Construct(input.Grid(), input.Root()));
  m_output_v.reset(input.Construct(input.Grid(), input.Root()));
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::setup_dims() {
  data_type_layer<TensorDataType>::setup_dims();

  // Get concatenate points for first parent layer
  auto output_dims = this->get_input_dims(0);
  if (m_concat_dim < 0
      || m_concat_dim >= (El::Int) output_dims.size()) {
    LBANN_ERROR(get_type()," layer \"",this->get_name(),"\" ",
                "has ",output_dims.size()," dimensions, ",
                "but attempted to concatenate along ",
                "dimension ",m_concat_dim);
  }
  m_concat_points.clear();
  m_concat_points.push_back(0);
  m_concat_points.push_back(output_dims[m_concat_dim]);

  // Get concatenation points for remaining parent layers
  for (int i = 1; i < this->get_num_parents(); ++i) {
    const auto& input_dims = this->get_input_dims(i);
    if (input_dims.size() != output_dims.size()
        || !std::equal(input_dims.begin(),
                       input_dims.begin() + m_concat_dim,
                       output_dims.begin())
        || !std::equal(input_dims.begin() + m_concat_dim + 1,
                       input_dims.end(),
                       output_dims.begin() + m_concat_dim + 1)) {
      std::stringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "expects input tensors with dimensions ";
      for (size_t j = 0; j < output_dims.size(); ++j) {
        err << (j > 0 ? " x " : "");
        if ((int) j == m_concat_dim) {
          err << "X";
        } else {
          err << output_dims[j];
        }
      }
      err << ", but parent layer "
          << "\"" << this->get_parent_layers()[i]->get_name() << "\" "
          << "outputs with dimensions ";
      for (size_t j = 0; j < input_dims.size(); ++j) {
        err << (j > 0 ? " x " : "") << input_dims[j];
      }
      LBANN_ERROR(err.str());
    }
    output_dims[m_concat_dim] += input_dims[m_concat_dim];
    m_concat_points.push_back(output_dims[m_concat_dim]);
  }

  // Update output dimensions
  this->set_output_dims(output_dims);

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::fp_setup_outputs(El::Int mini_batch_size) {
  const auto& num_inputs = this->get_num_parents();
  const auto& output_dims = this->get_output_dims();

  // Initialize output tensor
  auto& output = this->get_activations();
  output.Empty(false);
  if (num_inputs > 1) {
    output.AlignWith(this->get_prev_activations());
    output.Resize(this->get_output_size(), mini_batch_size);
  } else {
    El::LockedView(output, this->get_prev_activations());
    return;
  }

  // Divide output tensor into unit slices along concat dimension
  // Note: Each unit slice is divided into contiguous "unit blocks"
  const auto& output_num_unit_slices = output_dims[m_concat_dim];
  const auto& blocks_per_slice
    = (m_concat_dim > 0 ?
       std::accumulate(&output_dims[0], &output_dims[m_concat_dim],
                       1, std::multiplies<int>()) :
       1);
  const auto& unit_block_size
    = std::accumulate(output_dims.begin() + m_concat_dim + 1,
                      output_dims.end(),
                      1, std::multiplies<int>());
  const auto& output_block_stride = (output_num_unit_slices
                                     * unit_block_size);

  // Populate slices of output tensor with input tensors
  for (int i = 0; i < num_inputs; ++i) {
    const auto& input_dims = this->get_input_dims(i);
    auto& input = this->get_prev_activations(i);

    // Divide input tensor into unit slices
    const auto& input_num_unit_slices = input_dims[m_concat_dim];

    // Merge unit slices
    const auto& block_size = input_num_unit_slices * unit_block_size;
    const auto& output_block_offset = m_concat_points[i] * unit_block_size;

    // Populate output tensor one block at a time
    for (int block = 0; block < blocks_per_slice; ++block) {
      const auto& input_offset = block * block_size;
      const auto& output_offset = (output_block_offset
                                   + block * output_block_stride);
      El::LockedView(*m_input_v, input,
                     El::IR(input_offset, input_offset + block_size),
                     El::ALL);
      El::View(*m_output_v, output,
               El::IR(output_offset, output_offset + block_size),
               El::ALL);
      El::Copy(*m_input_v, *m_output_v);
    }

  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) {
  const auto& num_inputs = this->get_num_parents();
  const auto& output_dims = this->get_output_dims();

  // Divide output tensor into unit slices along concat dimension
  // Note: Each unit slice is divided into contiguous "unit blocks"
  const auto& output_num_unit_slices = output_dims[m_concat_dim];
  const auto& blocks_per_slice
    = (m_concat_dim > 0 ?
       std::accumulate(&output_dims[0], &output_dims[m_concat_dim],
                       1, std::multiplies<int>()) :
       1);
  const auto& unit_block_size
    = std::accumulate(output_dims.begin() + m_concat_dim + 1,
                      output_dims.end(),
                      1, std::multiplies<int>());
  const auto& output_block_stride = (output_num_unit_slices
                                     * unit_block_size);

  // Populate gradient w.r.t. input tensors
  const auto& gradient_wrt_output = this->get_prev_error_signals();
  for (int i = 0; i < num_inputs; ++i) {
    const auto& input_dims = this->get_input_dims(i);
    const auto& input_size = this->get_input_size(i);
    auto& gradient_wrt_input = this->get_error_signals(i);

    // Divide input tensor into unit slices
    const auto& input_num_unit_slices = input_dims[m_concat_dim];

    // Merge unit slices and get first contiguous output block
    const auto& block_size = input_num_unit_slices * unit_block_size;
    const auto& output_block_offset = m_concat_points[i] * unit_block_size;
    El::LockedView(*m_output_v, gradient_wrt_output,
                   El::IR(output_block_offset,
                          output_block_offset + block_size),
                   El::ALL);

    // Populate gradient w.r.t. input tensor one block at a time
    // Note: If there is only one block, the tensor can be a view
    if (blocks_per_slice > 1) {
      gradient_wrt_input.AlignWith(*m_output_v);
      gradient_wrt_input.Resize(input_size, mini_batch_size);
      for (int block = 0; block < blocks_per_slice; ++block) {
        const auto& input_offset = block * block_size;
        const auto& output_offset = (output_block_offset
                                     + block * output_block_stride);
        El::LockedView(*m_output_v, gradient_wrt_output,
                       El::IR(output_offset, output_offset + block_size),
                       El::ALL);
        El::View(*m_input_v, gradient_wrt_input,
                 El::IR(input_offset, input_offset + block_size),
                 El::ALL);
        El::Copy(*m_output_v, *m_input_v);
      }
    } else {
      El::LockedView(gradient_wrt_input, *m_output_v);
    }

  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::fp_compute() {
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::bp_compute() {
}

#ifndef LBANN_CONCATENATE_LAYER_INSTANTIATE
extern template class concatenate_layer<
  DataType, data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class concatenate_layer<
  DataType, data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class concatenate_layer<
  DataType, data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class concatenate_layer<
  DataType, data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_CONCATENATE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_TRANSFORM_CONCATENATE_HPP_INCLUDED
