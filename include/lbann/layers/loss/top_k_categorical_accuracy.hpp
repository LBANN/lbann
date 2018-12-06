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

#ifndef LBANN_LAYERS_LOSS_TOP_K_CATEGORICAL_ACCURACY_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_TOP_K_CATEGORICAL_ACCURACY_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

/** Top-k categorical accuracy layer.
 *  The two inputs are interpreted as prediction scores and
 *  ground-truth labels, respectively. An output is set to one if the
 *  corresponding label matches one of the top-k prediction scores and
 *  is otherwise zero. Each label is assumed to be a one-hot vector
 *  and ties in the top-k prediction scores are broken in favor of
 *  entries with smaller indices.
 *  @todo Gracefully handle case where label is not a one-hot vector.
 */
template <data_layout T_layout, El::Device Dev>
class top_k_categorical_accuracy_layer : public Layer {
public:

  top_k_categorical_accuracy_layer(lbann_comm *comm, El::Int k)
    : Layer(comm), m_k(k) {
    this->m_expected_num_parent_layers = 2;
  }

  top_k_categorical_accuracy_layer* copy() const override {
    return new top_k_categorical_accuracy_layer(*this);
  }
  std::string get_type() const override { return "top-k accuracy"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims({1});

    // Check that input dimensions are valid
    std::stringstream err;
    const auto& parents = get_parent_layers();
    const auto& dims0 = get_input_dims(0);
    const auto& dims1 = get_input_dims(1);
    if (dims0 != dims1) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "expects input tensors with identical dimensions, "
          << "but parent layer \"" << parents[0]->get_name() << "\" "
          << "outputs a tensor with dimensions ";
      for (size_t i = 0; i < dims0.size(); ++i) {
        err << (i > 0 ? " x " : "") << dims0[i];
      }
      err << " and parent layer \"" << parents[1]->get_name() << "\" "
          << "outputs a tensor with dimensions ";
      for (size_t i = 0; i < dims1.size(); ++i) {
        err << (i > 0 ? " x " : "") << dims1[i];
      }
      LBANN_ERROR(err.str());
    }

  }

  void fp_compute() override;

private:

  /** Parameter for top-k search. */
  const El::Int m_k;

};

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_TOP_K_CATEGORICAL_ACCURACY_HPP_INCLUDED
