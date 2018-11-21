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

#ifndef LBANN_LAYERS_GENERIC_TARGET_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_GENERIC_TARGET_LAYER_HPP_INCLUDED

#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/io/data_buffers/distributed_io_buffer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
class generic_target_layer : public Layer {
 public:
  generic_target_layer(lbann_comm *comm)
    : Layer(comm) {
    // Target layers have two parents, the layer that it will feed
    // back the response to, and the input layer where it gets the response
    m_expected_num_parent_layers = 2;
    // Target layers have no children
    m_expected_num_child_layers = 0;
  }

  generic_target_layer(const generic_target_layer& other)
    : Layer(other) {}

  generic_target_layer& operator=(const generic_target_layer& other) {
    Layer::operator=(other);

    return *this;
  }


  ~generic_target_layer() override {};

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::string s = get_topo_description();
    return std::string {} + " target_layer "
           + " dataLayout: " + this->get_data_layout_string(get_data_layout())
           + " (" + s + ")";
  }

  void fp_compute() override {
    // Differentiating objective function loss functions requires that
    // target layer error signal matrices are zero
    for (int i = 0; i < get_num_parents(); ++i) {
      const auto& input = get_prev_activations(i);
      auto& gradient_wrt_input = get_error_signals(i);
      El::Zeros(gradient_wrt_input, input.Height(), input.Width());
    }
  }

  void bp_compute() override {}

  bool update_compute() override {
    return true;
  }

  virtual const AbsDistMat& get_prediction() const { return get_prev_activations(0); }
  virtual const AbsDistMat& get_ground_truth() const { return get_prev_activations(1); }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_GENERIC_TARGET_LAYER_HPP_INCLUDED
