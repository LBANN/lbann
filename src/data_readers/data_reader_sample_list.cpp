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

#include "lbann/data_readers/data_reader_sample_list.hpp"
#include "lbann/data_readers/sample_list_impl.hpp"
#include "lbann/data_readers/sample_list_open_files_impl.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/vectorwrapbuf.hpp"

namespace lbann {

data_reader_sample_list::data_reader_sample_list(bool shuffle)
  : generic_data_reader(shuffle)
{}

data_reader_sample_list::data_reader_sample_list(
  const data_reader_sample_list& rhs)
  : generic_data_reader(rhs)
{
  copy_members(rhs);
}

data_reader_sample_list&
data_reader_sample_list::operator=(const data_reader_sample_list& rhs)
{
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}

void data_reader_sample_list::copy_members(const data_reader_sample_list& rhs)
{
  m_sample_list.copy(rhs.m_sample_list);
}

void data_reader_sample_list::shuffle_indices(rng_gen& gen)
{
  generic_data_reader::shuffle_indices(gen);
  m_sample_list.compute_epochs_file_usage(get_shuffled_indices(),
                                          get_mini_batch_size(),
                                          *m_comm);
}

void data_reader_sample_list::load()
{
  if (is_master()) {
    std::cout << "starting data_reader_sample_list::load()\n";
  }
  const std::string sample_list_file = get_data_sample_list();
  if (sample_list_file.empty()) {
    LBANN_ERROR("sample list was not specified.");
  }
  load_list_of_samples(sample_list_file);
}

void data_reader_sample_list::load_list_of_samples(
  const std::string sample_list_file)
{
  // load the sample list
  double tm1 = get_time();
  options* opts = options::get();

  // dah: I've not a clue what this next block does;
  //      is it a hack that should come out?
  if (this->m_keep_sample_order || opts->has_string("keep_sample_order")) {
    m_sample_list.keep_sample_order(true);
  }
  else {
    m_sample_list.keep_sample_order(false);
  }

  // Load the sample list
  if (opts->has_string("slosload_full_sample_list_once")) {
    std::vector<char> buffer;
    if (m_comm->am_trainer_master()) {
      load_file(sample_list_file, buffer);
    }
    m_comm->trainer_broadcast(m_comm->get_trainer_master(), buffer);

    vectorwrapbuf<char> strmbuf(buffer);
    std::istream iss(&strmbuf);

    m_sample_list.set_sample_list_name(sample_list_file);
    m_sample_list.load(iss, *(this->m_comm), true);
  }
  else {
    m_sample_list.load(sample_list_file, *(this->m_comm), true);
  }
  if (is_master()) {
    std::cout << "Time to load sample list '" << sample_list_file
              << "': " << get_time() - tm1 << std::endl;
  }

  // Merge all of the sample lists
  double tm3 = get_time();
  m_sample_list.all_gather_packed_lists(*m_comm);

  if (is_master()) {
    std::cout << "Time to gather sample list '" << sample_list_file
              << "': " << get_time() - tm3 << std::endl;
  }

  // Set base directory for your data.
  generic_data_reader::set_file_dir(m_sample_list.get_samples_dirname());
}

void data_reader_sample_list::load_list_of_samples_from_archive(
  const std::string& sample_list_archive)
{
  // load the sample list
  double tm1 = get_time();
  std::stringstream ss(sample_list_archive); // any stream can be used

  cereal::BinaryInputArchive iarchive(ss); // Create an input archive

  iarchive(m_sample_list); // Read the data from the archive
  double tm2 = get_time();

  if (is_master()) {
    std::cout << "Time to load sample list from archive: " << tm2 - tm1
              << std::endl;
  }
}
void data_reader_sample_list::open_file(size_t index_in,
                                        hid_t& file_handle_out,
                                        std::string& sample_name_out)
{
  const sample_t& s = m_sample_list[index_in];
  sample_name_out = s.second;
  sample_file_id_t id = s.first;
  m_sample_list.open_samples_file_handle(index_in);
  file_handle_out = m_sample_list.get_samples_file_handle(id);
  if (!m_sample_list.is_file_handle_valid(file_handle_out)) {
    LBANN_ERROR("file handle is invalid");
  }
}

void data_reader_sample_list::close_file(size_t index)
{
  m_sample_list.close_samples_file_handle(index);
}

} // end of namespace lbann
