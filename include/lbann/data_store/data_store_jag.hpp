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
////////////////////////////////////////////////////////////////////////////////

#ifndef __DATA_STORE_JAG_HPP__
#define __DATA_STORE_JAG_HPP__

#include "lbann_config.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "lbann/data_store/generic_data_store.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_hdf5.hpp"
#include "conduit/conduit_relay_mpi.hpp"
#include <unordered_map>

namespace lbann {

class data_reader_jag_conduit_hdf5;
typedef data_reader_jag_conduit_hdf5 conduit_reader;
//class data_reader_jag_conduit;
//typedef data_reader_jag_conduit conduit_reader;


class data_store_jag : public generic_data_store {
 public:

  //! ctor
  data_store_jag(generic_data_reader *reader, model *m);

  //! copy ctor
  data_store_jag(const data_store_jag&) = default;

  //! operator=
  data_store_jag& operator=(const data_store_jag&) = default;

  data_store_jag * copy() const override { return new data_store_jag(*this); }

  //! dtor
  ~data_store_jag() override;

  void setup() override;

  void set_reader(conduit_reader *r) { m_data_reader = r; }

  /// return the conduit node that contains the requested sample
  /// data_id; tid is ignored for now, since nodes are not thread-safe
  const conduit::Node & get_node(int data_id, int tid) const;

protected :

  conduit_reader *m_data_reader;

  /// fills in m_owner: std::unordered_map<int, int> m_owner
  /// which maps an index to the processor that owns the associated data;
  /// m_owner is in generic_data_store.hpp
  void build_index_owner() override;

  /// buffers for data that will be passed to the data reader's fetch_datum method
  /// std::unordered_map<int, std::vector<DataType>> m_my_minibatch_data;

  /// retrive data needed for passing to the data reader for the next epoch
  void exchange_data() override;

  /// returns, in "indices," the set of indices that processor "p"
  /// needs for the next epoch. Called by exchange_data
  //void get_indices(std::unordered_set<int> &indices, int p);

  /// This vector contains the Nodes that this processor owns
  std::vector<conduit::Node> m_data;

  /// fills in m_data (the data store)
  void populate_datastore();

  /// This vector contains Nodes that this processor needs for
  /// the current minibatch; this is filled in by exchange_data()
  std::vector<conduit::Node> m_minibatch_data;

  /// work space; used in exchange_data
  std::vector<conduit::Node> m_send_buffer;
  std::vector<conduit::Node> m_send_buffer_2;
  std::vector<MPI_Request> m_send_requests;
  std::vector<MPI_Request> m_recv_requests;
  std::vector<MPI_Status> m_status;
  std::vector<conduit::Node> m_recv_buffer;

  std::vector<int> m_outgoing_msg_sizes;
  std::vector<int> m_incoming_msg_sizes;

  /// m_sample_ownership[j] is the global index of the first sample that
  /// P_j owns. m_sample_ownership[j+1] - m_sample_ownership[j] is the 
  /// number of samples that P_j owns
  std::vector<int> m_sample_ownership;

  int m_num_samples;

  void testme();

  void build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out);
};

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT

#endif  // __DATA_STORE_JAG_HPP__
