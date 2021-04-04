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

#include "lbann/callbacks/callback.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/io/persist_impl.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/serialize.hpp"
#include <iomanip>
#include <lbann.pb.h>
#include <queue>
#include <string>
#include <unistd.h>
#include <unordered_set>

namespace lbann {

//******************************************************************************
// Execution context
//******************************************************************************

execution_context::execution_context(trainer& trainer,
                                     training_algorithm& training_algorithm,
                                     execution_mode mode)
  : m_trainer(&trainer), m_training_algorithm(&training_algorithm),
    m_comm(trainer.get_comm()), m_execution_mode(mode),
    m_terminate_training(false)
{}

template <class Archive> void execution_context::serialize(Archive& ar)
{
  ar(CEREAL_NVP(m_execution_mode),
     CEREAL_NVP(m_terminate_training),
     CEREAL_NVP(m_step));
}

////////////////////////////////////////////////////////////
// Training_Algorithm state
////////////////////////////////////////////////////////////

// observer_ptr<thread_pool> training_algorithm::get_io_thread_pool() {
//   return m_trainer->get_io_thread_pool();
// }

thread_pool& execution_context::get_io_thread_pool() const
{
  return m_trainer->get_io_thread_pool();
}

////////////////////////////////////////////////////////////
// Checkpointing
////////////////////////////////////////////////////////////

void execution_context::save_to_checkpoint_shared(persist& p)
{
  if (get_comm().am_trainer_master()) {
    write_cereal_archive<execution_context>(*this,
                                            p,
                                            get_execution_mode(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                            "_ctx.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                            "_ctx.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
    );
  }
  return;
}

void execution_context::load_from_checkpoint_shared(persist& p)
{
  load_from_shared_cereal_archive<execution_context>(*this,
                                                     p,
                                                     get_execution_mode(),
                                                     get_comm(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                                     "_ctx.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                                     "_ctx.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return;
}

void execution_context::save_to_checkpoint_distributed(persist& p)
{
  write_cereal_archive<execution_context>(*this,
                                          p,
                                          get_execution_mode(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                          "_ctx.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                          "_ctx.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return;
}

void execution_context::load_from_checkpoint_distributed(persist& p)
{
  read_cereal_archive<execution_context>(*this,
                                         p,
                                         get_execution_mode(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                         "_ctx.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                         "_ctx.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return;
}

} // namespace lbann

#define LBANN_CLASS_NAME execution_context
#include <lbann/macros/register_class_with_cereal.hpp>
