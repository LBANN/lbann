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

#ifndef SOFTPLUS_HPP_INCLUDED
#define SOFTPLUS_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/** Softplus activation function.
 *  This is a smooth approximation of the ReLU. See
 *  https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
template <data_layout T_layout, El::Device Dev>
class softplus_layer : public entrywise_activation_layer {
public :
  softplus_layer(lbann_comm *comm) : entrywise_activation_layer(comm) {}
  softplus_layer* copy() const override { return new softplus_layer(*this); }
  std::string get_type() const override { return "softplus"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:
  DataType activation(DataType x) const override {
    // Warning: Not numerically stable.
    // Better approach is to determine a threshold so that for large x,
    // softplus(x) ~= x and for small x, softplus(x) ~= exp(x).
    return std::log1p(std::exp(x));
  }
  DataType activation_derivative(DataType x) const override {
    return 1 / (DataType(1) + std::exp(-x));
  }
};

} // namespace lbann

#endif // SOFTPLUS_HPP_INCLUDED
