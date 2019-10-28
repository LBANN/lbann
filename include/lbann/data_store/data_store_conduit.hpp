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
//
////////////////////////////////////////////////////////////////////////////////

#ifndef __DATA_STORE_CONDUIT_HPP__
#define __DATA_STORE_CONDUIT_HPP__

#include "lbann_config.hpp"

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "conduit/conduit_node.hpp"
#include <unordered_map>
#include <unordered_set>
#include <mutex>


namespace lbann {

// support for encoding data_id in conduit::Node, used by
// conduit_data_store and associated code
#define LBANN_SAMPLE_ID_PAD 9
#define LBANN_DATA_ID_STR(data_id) pad(std::to_string(data_id), LBANN_SAMPLE_ID_PAD, '0')

class generic_data_reader;

class data_store_conduit {

 public:

  //! ctor
  data_store_conduit(generic_data_reader *reader);

  //! copy ctor
  data_store_conduit(const data_store_conduit&);

  //! copy / split ctor
  data_store_conduit(const data_store_conduit&, const std::vector<int>&);

  //! operator=
  data_store_conduit& operator=(const data_store_conduit&);

  data_store_conduit * copy() const { return new data_store_conduit(*this); }

  //! dtor
  ~data_store_conduit();

  /// required when the copy ctor is used to construct a validation set
  void set_data_reader_ptr(generic_data_reader *reader);

  //! convenience handle
  void set_shuffled_indices(const std::vector<int> *indices);

  /** @brief Returns the number of samples summed over all ranks */
  size_t get_num_global_indices() const;

  void setup(int mini_batch_size);

  void preload_local_cache();

  void check_mem_capacity(lbann_comm *comm, const std::string sample_list_file, size_t stride, size_t offset);

  /// returns the conduit node
  const conduit::Node & get_conduit_node(int data_id) const;

  /// if 'already_have = true' then the passed 'node' was obtained by a call to
  /// get_empty_node(). In some operating modes this saves us from copying the node
  void set_conduit_node(int data_id, conduit::Node &node, bool already_have = false);

  void set_preloaded_conduit_node(int data_id, const conduit::Node &node);
  void spill_preloaded_conduit_node(int data_id, const conduit::Node &node);

  const conduit::Node & get_random_node() const;

  const conduit::Node & get_random_node(const std::string &field) const;

  /// returns an empty node
  conduit::Node & get_empty_node(int data_id);

  void set_is_preloaded(); 

  bool is_preloaded() { return m_preload; }

  void set_explicit_loading(bool flag) { m_explicit_loading = flag; }

  bool is_explicitly_loading() { return m_explicit_loading; }

  /// fills in m_owner, which maps index -> owning processor
  void build_preloaded_owner_map(const std::vector<int>& per_rank_list_sizes);

  /// Removed nodes corresponding from the indices vector from the data store
  void purge_unused_samples(const std::vector<int>& indices);

  /// Recompact the nodes because they are not copied properly when instantiating
  /// using the copy constructor
  void compact_nodes();

  /// returns the processor that owns the data associated
  /// with the index
  int get_index_owner(int idx);

  bool is_local_cache() const { return m_is_local_cache; }

  void exchange_mini_batch_data(size_t current_pos, size_t mb_size); 

  void set_node_sizes_vary() { m_node_sizes_vary = true; }

  bool has_conduit_node(int data_id) const;

  /// only used for debugging; pass --debug on cmd line to get
  /// each data store to print to a different file. This is made
  /// public so data readers can also print to the file
  std::ofstream *m_debug = nullptr;
  std::ofstream *m_profile = nullptr;

  /// for use during development and debugging
  int get_data_size() { return m_data.size(); }

  /// made public for debugging during development
  void copy_members(const data_store_conduit& rhs, const std::vector<int>& = std::vector<int>());

  /** @brief Closes then reopens the debug logging file
   *
   * Debug logging is enabled on all ranks via the cmd line flag: --data_store_debug
   */
  void flush_debug_file(); 


  /** @brief Closes then reopens the profile logging file
   *
   * Profile logging is enabled on P_0 via the cmd line flag: --data_store_profile
   */
  void flush_profile_file(); 

  /** @brief Writes object's state to file */
  void write_checkpoint(std::string dir_name);
  
  /** @brief Loads object's state from file */
  void load_checkpoint(std::string dir_name, generic_data_reader *reader = nullptr);

private :

  /** @brief The number of samples that this processor owns */
  size_t m_my_num_indices = 0;

  /** @brief if true, then we are spilling (offloading) samples to disk */
  bool m_spill = false;

  /** @brief if true, then all samples have been spilled */
  bool m_is_spilled = false;

  /** During spilling, the conduit file pathnames are written to this file */
  std::ofstream m_metadata;

  /** @brief Base directory for spilling (offloading) conduit nodes */
  std::string m_spill_dir_base;

  /** @brief Used to form the directory path for spilling conduit nodes */
  int m_cur_spill_dir_integer = -1;

  /** @brief @brief Current directory for spilling (writing to file) conduit nodes 
   *
   * m_cur_spill_dir = m_spill_dir_base/<m_cur_spill_dir_integer>
   */
  std::string m_cur_spill_dir;

  /** @brief The directory to use for testing checkpointing
   *
   * Testing is activated by passing the cmd flag: --data_store_test_checkpoint=<dir>
   */
  std::string m_test_dir;

  /** @brief Contains the number of conduit nodes that have been written to m_cur_dir
   *
   * When m_num_files_in_cur_spill_dir == m_max_files_per_directory,
   * m_cur_spill_dir_integer is incremented and a new m_cur_dir is created
   */
  int m_num_files_in_cur_spill_dir;

  /** @brief maps data_id to m_m_cur_spill_dir_integer. */
  std::unordered_map<int, int> m_spilled_nodes;

  /// used in set_conduit_node(...)
  std::mutex m_mutex;

  /// for use in local cache mode
  char *m_mem_seg = 0;
  size_t m_mem_seg_length = 0;
  std::string m_seg_name;

  const std::string m_debug_filename_base = "debug";
  std::string m_debug_filename;

  const std::string m_profile_filename_base = "data_store_profile";
  std::string m_profile_filename;

  bool m_was_loaded_from_file = false;
  const std::string m_cereal_fn = "data_store_cereal";

  /// used in spill_to_file
  /// (actually, conduit::Node.save() writes both a
  ///  json file and a binary file, so double this number
  const int m_max_files_per_directory = 500;

  //===========================================================
  // timers for profiling exchange_data
  //===========================================================

  // applicable to imagenet; NA for JAG
  double m_exchange_sample_sizes_time = 0;

  // time from beginning of exchange_data_by_sample to wait_all
  double m_start_snd_rcv_time = 0;

  // time for wait_all
  double m_wait_all_time = 0;

  // time to unpack nodes received from other ranks
  double m_rebuild_time = 0;

  // total time for exchange_mini_batch_data
  double m_exchange_time = 0; 

  // sanity check: 
  //   m_start_snd_rcv_time + m_wait_all_time + m_rebuild_time
  // should be only slightly less than m_exchange_time;
  // Note that, for imagenet, the first call to exchange_data_by_sample
  // involves additional communication for exchanging sample sizes
 
  //===========================================================
  // END: timers for profiling exchange_data
  //===========================================================

  int m_cur_epoch = 0;

  bool m_is_setup = false;

  /// set to true if data_store is preloaded
  bool m_preload = false;

  /// set to true if data_store is being explicitly loaded
  //VBE: please explain what this means!
  bool m_explicit_loading = false;

  /// The size of the mini-batch that was used to calculate ownership
  /// of samples when building the owner map.  This size has to be
  /// used consistently when computing the indices that will be sent
  /// and received.
  int m_owner_map_mb_size = 0;

  /// size of a compacted conduit::Node that contains a single sample
  int m_compacted_sample_size = 0;

  bool m_is_local_cache = false;

  bool m_node_sizes_vary = false;

  /// used in exchange_data_by_sample, when sample sizes are non-uniform
  bool m_have_sample_sizes = false;

  generic_data_reader *m_reader;

  lbann_comm *m_comm = nullptr;

  /// convenience handles
  bool m_world_master;
  bool m_trainer_master;
  int  m_rank_in_trainer;
  int  m_rank_in_world = -1; // -1 for debugging 
  int  m_np_in_trainer;

  /** @brief Maps an index to the processor that owns the associated data
   *
   * Must be mutable since rhs.m_owner may be modified in copy_members,
   * in which rhs is const.
   */ 
  //TODO: make undoredered map; for development want map() for ordered printing
  mutable std::map<int, int> m_owner;

  /// convenience handle
  const std::vector<int> *m_shuffled_indices;

  /** @brief Contains the conduit nodes that are "owned" by this rank
   *
   * Map data_id -> conduit::Node.
   * Must be mutable since rhs.m_owner may be modified in copy_members,
   * in which rhs is const.
   */ 
  mutable std::unordered_map<int, conduit::Node> m_data;

  /// Contains the list of data IDs that will be received
  std::vector<int> m_recv_data_ids;
  std::unordered_map<int, int> m_recv_sample_sizes;

  /// This vector contains Nodes that this processor needs for
  /// the current minibatch; this is filled in by exchange_data()
  std::unordered_map<int, conduit::Node> m_minibatch_data;

  /// work space; used in exchange_data
  std::vector<conduit::Node> m_send_buffer;
  std::vector<conduit::Node> m_send_buffer_2;
  std::vector<El::mpi::Request<El::byte>> m_send_requests;
  std::vector<El::mpi::Request<El::byte>> m_recv_requests;
  std::vector<conduit::Node> m_recv_buffer;
  std::vector<size_t> m_outgoing_msg_sizes;
  std::vector<size_t> m_incoming_msg_sizes;

  /// for use when conduit Nodes have non-uniform size, e.g, imagenet
  std::unordered_map<int, size_t> m_sample_sizes;

  /// maps processor id -> set of indices (whose associated samples)
  /// this proc needs to send. (formerly called "proc_to_indices);
  /// this is filled in by build_indices_i_will_send()
  std::vector<std::unordered_set<int>> m_indices_to_send;

  /// maps processor id -> set of indices (whose associated samples)
  /// this proc needs to recv from others. (formerly called "needed")
  std::vector<std::unordered_set<int>> m_indices_to_recv;

  /// offset at which the raw image will be stored in a shared memory segment;
  /// for use in local cache mode; maps data_id to offset
  std::unordered_map<int,size_t> m_image_offsets;

  //=========================================================================
  // methods follow 
  //=========================================================================

  void exchange_data_by_sample(size_t current_pos, size_t mb_size);

  void setup_data_store_buffers();

  /// called by exchange_data
  void build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out);

  /// fills in m_owner, which maps index -> owning processor
  void exchange_owner_maps();

  /// for use when conduit Nodes have non-uniform size, e.g, imagenet
  void exchange_sample_sizes();

  /// fills in m_indices_to_send and returns the number of samples
  /// that will be sent
  int build_indices_i_will_send(int current_pos, int mb_size);

  /// fills in m_indices_to_recv and returns the number of samples
  /// that will be received
  int build_indices_i_will_recv(int current_pos, int mb_size);

  void error_check_compacted_node(const conduit::Node &nd, int data_id);

  /// Currently only used for imagenet. On return, 'sizes' maps a sample_id to image size, and indices[p] contains the sample_ids that P_p owns
  /// for use in local cache mode
  void get_image_sizes(std::unordered_map<int,size_t> &sizes, std::vector<std::vector<int>> &indices);

  /// fills in m_image_offsets for use in local cache mode
  void compute_image_offsets(std::unordered_map<int,size_t> &sizes, std::vector<std::vector<int>> &indices);

  /// for use in local cache mode
  void allocate_shared_segment(std::unordered_map<int,size_t> &sizes, std::vector<std::vector<int>> &indices);

  /// for use in local cache mode
  void read_files(std::vector<char> &work, std::unordered_map<int,size_t> &sizes, std::vector<int> &indices);

  /// for use in local cache mode
  void build_conduit_nodes(std::unordered_map<int,size_t> &sizes);

  /// for use in local cache mode
  void exchange_images(std::vector<char> &work, std::unordered_map<int,size_t> &image_sizes, std::vector<std::vector<int>> &indices); 

  /// for use in local cache mode
  void fillin_shared_images(const std::vector<char> &images, size_t offset);

  /** @brief For testing during development
   *
   * At the beginning of the 2nd epoch, calls write_checkpoint(), 
   * clears some variables, calls load_checkpoint then continues. 
   * To activate this test use cmd flag: --data_store_test_checkpoint=
   */ 
  void test_checkpoint(const std::string&);

  /** @brief Called by test_checkpoint */
  void print_variables();

  /** @brief Called by test_checkpoint */
  void print_partial_owner_map(int n);

  std::string get_conduit_dir() const;
  std::string get_cereal_fn() const;
  std::string get_metadata_fn() const;


  /** @brief Creates the directory if it does not already exist */
  void make_dir_if_it_doesnt_exist(const std::string &dir); 

  /** @brief Writes conduit node to file */
  void spill_conduit_node(const conduit::Node &node, int data_id);

  /** @brief Loads conduit nodes from file into m_data */
  void load_spilled_conduit_nodes();

  /** @brief Creates directory structure, opens metadata file for output, etc
   *
   * This method is called for both --data_store_spill and 
   * --data_store_test_checkpoint 
   */
  void setup_spill(const std::string &dir);

  /** @brief Saves this object's state to file
   *
   * Here, "state" is all data, except for conduit nodes, that is
   * needed to reload from checkpoint
   */
  void save_state();

  /** @brief Optionally open debug and profiling files
   *
   * A debug file is opened for every <rank, data reader role> pair;
   * files are opened if the cmd flag --data_store_debug is passed.
   * A profiling file is opened only be <world_master, data reader role>
   * pairs; files are opened if the cmd flag --data_store_profile is passed.
   */ 
  void open_informational_files();

  /** @brief Creates a directory for spilling conduit nodes */
  void open_next_conduit_spill_directory();

  //=========================================================================
  // functions and templates for optional profiling and debug files follow
  //=========================================================================

  void PROFILE() { 
    if (!m_profile) {
      return;
    }
    (*m_profile) << std::endl; 
    flush_profile_file();
  }

  template <typename T, typename... Types>
  void PROFILE(T var1, Types... var2) {
    if (!m_world_master) {
      return;
    }
    if (!m_profile) {
      return;
    }
    (*m_profile) << var1 << " ";
    PROFILE(var2...) ;
  }

  void DEBUG() { 
    if (!m_debug) {
      return;
    }
    (*m_debug) << std::endl; 
    flush_debug_file();
  }

  template <typename T, typename... Types>
  void DEBUG(T var1, Types... var2) {
    if (!m_debug) {
      return;
    }
    (*m_debug) << var1 << " ";
    DEBUG(var2...) ;
  }

};

}  // namespace lbann


#endif  // __DATA_STORE_JAG_HPP__
