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

#ifndef ELU_HPP_INCLUDED
#define ELU_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/** Exponential linear unit.

 *  Tries to speed up learning by pushing the mean of activations more
 *  towards zero by allowing negative values. Helps avoid the need for
 *  batch normalization. See:
 *  Djork-Arne Clevert, Thomas Unterthiner, and Sepp Hochreiter "Fast
 *  and Accurate Deep Network Learning by Exponential Linear Units
 *  (ELUs)" ICLR 2016.
 */
template <data_layout T_layout, El::Device Dev>
class elu_layer : public entrywise_activation_layer {
 public:
  /**
   * alpha controls the value to which the ELU saturates for negative inputs.
   * alpha must be >= 0.
   * If alpha = 0, this turns into a ReLU.
   * Paper uses alpha = 1.0 as a good starting point.
   */
  elu_layer(lbann_comm *comm,
            DataType alpha = DataType(1.0))
    : entrywise_activation_layer(comm), m_alpha(alpha) {}
  elu_layer* copy() const override { return new elu_layer(*this); }
  std::string get_type() const override { return "ELU"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:
  DataType activation(DataType x) const override {
    return (x > DataType(0)) ? x : (m_alpha * std::expm1(x));
  }
  DataType activation_derivative(DataType x) const override {
    return (x > DataType(0)) ? DataType(1) : (m_alpha * std::expm1(x) + m_alpha);
  }
 private:
  DataType m_alpha;
};

} // namespace lbann

#endif // ELU_HPP_INCLUDED
