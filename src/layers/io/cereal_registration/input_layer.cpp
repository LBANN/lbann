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

#include <cereal/types/polymorphic.hpp>
#include <lbann/layers/io/input_layer.hpp>

#define LBANN_COMMA ,
#define LBANN_REGISTER_LAYER_WITH_CEREAL_BASE(TYPE, LAYOUT, DEVICE) \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                       \
    ::lbann::input_layer<TYPE LBANN_COMMA ::lbann::data_layout::LAYOUT LBANN_COMMA DEVICE>, \
    "input_layer(" #TYPE "," #LAYOUT "," #DEVICE ")")

#define LBANN_REGISTER_LAYER_WITH_CEREAL(TYPE, DEVICE)                  \
  LBANN_REGISTER_LAYER_WITH_CEREAL_BASE(                                \
    TYPE, DATA_PARALLEL, DEVICE)

#define PROTO_DEVICE(T, D)                              \
  LBANN_REGISTER_LAYER_WITH_CEREAL(T, D)
#include <lbann/macros/instantiate_device.hpp>
