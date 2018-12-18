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

#ifndef LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

// Forward declaration.
class lbann_callback_imcomm;

/** @brief Transpose of the convolution layer.
 *  @todo Rename to "transposed_convolution_layer".
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class deconvolution_layer : public base_convolution_layer<Dev> {
private:

  friend class lbann_callback_imcomm;

public:

  deconvolution_layer(lbann_comm *comm,
                      int num_data_dims,
                      int num_output_channels,
                      int conv_dim,
                      int pad,
                      int stride,
                      int dilation,
                      int groups,
                      bool has_bias = true)
    : deconvolution_layer(comm,
                          num_data_dims,
                          num_output_channels,
                          std::vector<int>(num_data_dims, conv_dim),
                          std::vector<int>(num_data_dims, pad),
                          std::vector<int>(num_data_dims, stride),
                          std::vector<int>(num_data_dims, dilation),
                          groups,
                          has_bias) {}

  deconvolution_layer(lbann_comm *comm,
                      int num_data_dims,
                      int num_output_channels,
                      std::vector<int> conv_dims,
                      std::vector<int> pads,
                      std::vector<int> strides,
                      std::vector<int> dilations,
                      int groups,
                      bool has_bias = true)
    : base_convolution_layer<Dev>(comm,
                                  num_data_dims,
                                  num_output_channels,
                                  conv_dims,
                                  pads,
                                  strides,
                                  dilations,
                                  groups,
                                  has_bias) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "convolution only supports DATA_PARALLEL");

  }

  deconvolution_layer* copy() const override { return new deconvolution_layer(*this); }

  std::string get_type() const override { return "deconvolution"; }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  void setup_dims() override {
    base_convolution_layer<Dev>::setup_dims();
    std::stringstream err;

    // Get tensor dimensions
    const auto& input_dims = this->get_input_dims();
    auto output_dims = input_dims;
    const auto input_channels = input_dims[0];
    const auto output_channels = this->m_kernel_dims[0];

    // Check for unsupported features
    /// @todo Implement dilated and grouped deconvolution
    if (std::any_of(this->m_dilations.begin(),
                    this->m_dilations.end(),
                    [] (int d) { return d != 1; })) {
      err << this->get_type() << " layer "
          << "\"" << this->get_name() << "\" "
          << "has non-unit dilations (";
      for (size_t i = 0; i < this->m_dilations.size(); ++i) {
        err << (i > 0 ? ", " : "") << this->m_dilations[i];
      }
      err << ")";
      LBANN_ERROR(err.str());
    }
    if (this->m_num_groups != 1) {
      err << this->get_type() << " layer "
          << "\"" << this->get_name() << "\" "
          << "has non-unit groups "
          << "(" << this->m_num_groups << ")";
      LBANN_ERROR(err.str());
    }

    // Check that number of groups is valid
    if (this->m_num_groups < 1) {
      err << this->get_type() << " layer "
          << "\"" << this->get_name() << "\" "
          << "has " << this->m_num_groups << " groups";
      LBANN_ERROR(err.str());
    } else if (input_channels % this->m_num_groups != 0
               || output_channels % this->m_num_groups != 0) {
      err << this->get_type() << " layer "
          << "\"" << this->get_name() << "\" has "
          << input_channels << " input channels, "
          << output_channels << " output channels, and "
          << this->m_num_groups << " groups "
          << "(groups must evenly divide "
          << "the input channels and output channels)";
      LBANN_ERROR(err.str());
    }

    // Initialize convolution kernel dimensions
    // Note: Unlike the convolutional kernel, the previous layer's
    // number of channels is now the leading position -- keep in mind
    // that deconvolution is the transpose of a convolution.
    this->m_kernel_dims.insert(this->m_kernel_dims.begin(),
                               input_channels / this->m_num_groups);
    this->m_kernel_size = std::accumulate(this->m_kernel_dims.begin(),
                                          this->m_kernel_dims.end(),
                                          1, std::multiplies<int>());
    if (this->m_kernel_dims.size() != input_dims.size() + 1) {
      err << this->get_type() << " layer "
          << "\"" << this->get_name() << "\" has a ";
      for (size_t i = 0; i < input_dims.size(); ++i) {
        err << (i > 0 ? " x " : "") << input_dims[i];
      }
      err << " input tensor and a ";
      for (size_t i = 0; i < this->m_kernel_dims.size(); ++i) {
        err << (i > 0 ? " x " : "") << this->m_kernel_dims[i];
      }
      err << " convolution kernel";
      LBANN_ERROR(err.str());
    }

    // Initialize output tensor dimensions
    /// @todo Dilated deconvolution
    output_dims[0] = output_channels;
    for (size_t i = 0; i < output_dims.size() - 1; ++i) {
      const auto& input_dim = input_dims[i+1];
      const auto& kernel_dim = this->m_kernel_dims[i+2];
      const auto& stride = this->m_strides[i];
      const auto& pad = this->m_pads[i];
      // const auto& dilation = this->m_dilations[i];
      output_dims[i+1] = (input_dim-1) * stride + kernel_dim - 2 * pad;
    }
    this->set_output_dims(output_dims);

  }

protected:

  void fp_compute() override {
    if(this->using_gpus()) {
      base_convolution_layer<Dev>::apply_transposed_convolution_cudnn(true);
      base_convolution_layer<Dev>::apply_bias_cudnn();
    } else {
      base_convolution_layer<Dev>::apply_transposed_convolution_im2col(true);
      base_convolution_layer<Dev>::apply_bias_cpu();
    }
  }

  void bp_compute() override {
    if(this->using_gpus()) {
      base_convolution_layer<Dev>::compute_gradients_cudnn(true);
      base_convolution_layer<Dev>::apply_convolution_cudnn(false);
    } else {
      base_convolution_layer<Dev>::compute_gradients_im2col(true);
      base_convolution_layer<Dev>::apply_convolution_im2col(false);
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED
