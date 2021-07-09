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

#include "lbann/comm_impl.hpp"
#include "lbann/callbacks/perturb_weights.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include <callbacks.pb.h>

#include <algorithm>

namespace lbann {
namespace callback {

perturb_weights::perturb_weights(std::string output_name,
                           El::Int batch_interval)
  : callback_base(batch_interval),
    m_output_name(std::move(output_name)) {
}

perturb_weights::perturb_weights()
  : perturb_weights("",0)
{}

template <class Archive>
void perturb_weights::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_output_name));
}

void perturb_weights::setup(model* m) {
   weights* m_output = nullptr;
   
   for (auto* w : m->get_weights()) {
      if(w->get_name() == m_output_name){
        m_output = w;
        break;
      }
   }
    if (m_output == nullptr) {
      LBANN_ERROR("Current implementation of callback \"", name(), "\" "
                  "requires a weight object to perturb");
    }
}

void perturb_weights::on_batch_begin(model* m) {
  const auto& c = m->get_execution_context();
  weights* m_output = nullptr;

  for (auto* w : m->get_weights()) {
      if(w->get_name() == m_output_name){
        m_output = w;
        break;
      }
   }  
  
  if (m_output != nullptr && 
      c.get_step() % m_batch_interval == 0 &&
      c.get_step() >= 0) {
    perturb(*m);
  }
}

void perturb_weights::perturb(model& m){

  auto* comm = m.get_comm();

  // Useful constants
  constexpr DataType zero = 0;
  constexpr DataType one = 1;
  constexpr DataType lower = 0.3;
  constexpr DataType upper = 0.7;
  
  // RNG
  auto& gen = get_generator();
  std::normal_distribution<DataType> dist(zero, one);


  for (auto* w : m.get_weights()) {
    if (w == nullptr) {
    	LBANN_ERROR("callback \"", name(), "\" "
                  "got a weights pointer that is a null pointer");
    }

    // Check layer name
    if(w->get_name() == m_output_name) {
	auto& values = w->get_values();
	auto& new_values = dynamic_cast<El::AbstractDistMatrix<DataType>&>(values);

	auto& local_values = new_values.Matrix();
	El::Matrix<DataType,El::Device::CPU> temp;
	El::Copy(local_values, temp);

	// Perturb weights on master process		
	if (comm->am_trainer_master()) {
		for (auto i = 0; i < temp.Height(); i++){		

			
			// perturb				
			auto val = temp.Get(i,0);
			auto perturbed_val = val;
			
			perturbed_val += dist(gen); // dist is a std::normal_distribution
			perturbed_val = std::min(std::max(perturbed_val, lower), upper);
			
			temp.Set(i,0,perturbed_val);
				
			El::Copy(temp, local_values); 
			  
			std::cout << "Trainer [ " << m.get_comm()->get_trainer_rank() << " ], Step " << m.get_execution_context().get_step();
			std::cout << " Weight " << i << ": " << val << " Perturbed weight  " <<  perturbed_val << std::endl;

		}
	}

	// Communicate new weight from trainer master processes
	El::Broadcast(new_values, comm->get_trainer_comm(), 0);
		
	// Update weight
	auto& out_w = dynamic_cast<data_type_weights<DataType>&>(*w);	
	out_w.set_values(new_values);		

	break;
    }
  } 
}


std::unique_ptr<callback_base>
build_perturb_weights_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackPerturbWeights&>(proto_msg);
  return make_unique<perturb_weights>(
    params.output_name(),
    params.batch_interval());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::perturb_weights
#include <lbann/macros/register_class_with_cereal.hpp>
