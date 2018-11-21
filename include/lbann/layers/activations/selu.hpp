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

#ifndef SELU_HPP_INCLUDED
#define SELU_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/** SELU: scaled exponential linear unit.
 *  See: Klambauer et al. "Self-Normalizing Neural Networks", 2017.
 *  https://arxiv.org/abs/1706.02515
 *  By default, this assumes the goal is to normalize to 0 mean/unit
 *  variance. To accomplish this, you should also normalize input to 0
 *  mean/unit variance (z-score), initialize with 0 mean, 1/n variance
 *  (He), and use the SELU dropout.
 */
template <data_layout T_layout, El::Device Dev>
class selu_layer : public entrywise_activation_layer {
 public:
  selu_layer(lbann_comm *comm,
             DataType alpha = DataType(1.6732632423543772848170429916717),
             DataType scale = DataType(1.0507009873554804934193349852946))
    : entrywise_activation_layer(comm), m_alpha(alpha), m_scale(scale) {}
  selu_layer* copy() const override { return new selu_layer(*this); }
  std::string get_type() const override { return "SELU"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  std::string get_description() const override {
    return std::string {}
      + " selu" + " alpha: " + std::to_string(m_alpha) + " scale: "
      + std::to_string(m_scale) + " dataLayout: "
      + this->get_data_layout_string(get_data_layout());
  }

 protected:
  DataType activation(DataType x) const override {
    return (x >= DataType(0) ?
            m_scale * x :
            m_scale * (m_alpha * std::expm1(x)));
  }
  DataType activation_derivative(DataType x) const override {
    return (x >= DataType(0) ?
            m_scale :
            m_scale * m_alpha * std::exp(x));
  }
 private:
  /** Alpha parameter for the ELU. */
  DataType m_alpha;
  /** Scaling parameter for the result of the ELU. */
  DataType m_scale;
};

} // namespace lbann

#endif // SELU_HPP_INCLUDED
