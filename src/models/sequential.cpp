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
//
// sequential .hpp .cpp - Sequential neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/sequential.hpp"
#include <unordered_set>

namespace lbann {

sequential_model::sequential_model(lbann_comm *comm,
                                   int mini_batch_size,
                                   objective_function *obj_fn,
                                   optimizer* default_optimizer)
  : model(comm, mini_batch_size, obj_fn, default_optimizer) {}

void sequential_model::setup_layer_topology() {

  // Set up parent/child relationships between adjacent layers
  for (size_t i = 1; i < m_layers.size(); ++i) {
    m_layers[i]->add_parent_layer(m_layers[i-1]);
  }
  for (size_t i = 0; i < m_layers.size() - 1; ++i) {
    m_layers[i]->add_child_layer(m_layers[i+1]);
  }

  // Setup layer graph
  model::setup_layer_topology();

  // Make sure that execution order is valid
  std::set<int> nodes;
  std::map<int,std::set<int>> edges;
  construct_layer_graph(nodes, edges);
  if (!graph::is_topologically_sorted(nodes, edges)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "layer execution order is not topologically sorted";
    throw lbann_exception(err.str());
  }

  freeze_layers_under_frozen_surface();
}


void sequential_model::write_proto(lbann_data::Model* proto) {

  model::write_proto(proto);
  //Add layers
  if (m_comm->am_world_master()) {
    proto->set_name(name());
    for(size_t l = 0; l < m_layers.size(); l++) {
      auto layer_proto = proto->add_layer();
      m_layers[l]->write_proto(layer_proto);
    }
  }
}

}  // namespace lbann
