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


#include "lbann/data_readers/data_reader_python.hpp"
#ifdef LBANN_HAS_PYTHON
#include <cstdio>

namespace lbann {

namespace python {

std::unique_ptr<manager> manager::m_instance;

manager& manager::get_instance() {
  if (m_instance == nullptr) { create(); }
  return *m_instance;
}

void manager::create() {
  m_instance.reset(new manager());
}

void manager::destroy() {
  m_instance.reset(nullptr);
}

manager::manager() {
  if (!Py_IsInitialized()) {
    Py_Initialize();
  }
  if (!Py_IsInitialized()) {
    LBANN_ERROR("error creating embedded Python session");
  }
}

manager::~manager() {
  if (Py_IsInitialized()) {
    Py_Finalize();
  }
}

void manager::check_error(bool force_error) const {
  if (force_error || PyErr_Occurred()) {

    // Get error information from Python session
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);

    // Construct error message
    std::ostringstream err;
    err << "detected Python error";
    if (value != nullptr) {
      const char* msg = PyUnicode_AsUTF8(value);
      if (msg != nullptr) {
        err << " (" << msg << ")";
      }
    }
    if (traceback != nullptr) {
      auto tb_module = PyImport_ImportModule("traceback");
      auto tb_message = PyObject_CallMethod(tb_module,
                                            "format_exc",
                                            nullptr);
      const char* tb_str = PyUnicode_AsUTF8(tb_message);
      if (tb_str != nullptr) {
        err << "\n\n" << tb_str;
      }
      Py_XDECREF(tb_module);
      Py_XDECREF(tb_message);
    }

    // Clean up and throw exception
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    LBANN_ERROR(err.str());

  }
}

manager::mutex_guard_type manager::get_mutex_guard() {
  return mutex_guard_type(m_mutex);
}

object::object(PyObject* ptr) : m_ptr(ptr) {
  if (Py_IsInitialized() && PyErr_Occurred()) {
    manager::get_instance().check_error();
  }
}

object::object(std::string val)
  : object(PyUnicode_FromStringAndSize(val.c_str(), val.size())) {}
object::object(El::Int val) : object(PyLong_FromLong(val)) {}
object::object(DataType val) : object(PyFloat_FromDouble(val)) {}

object::object(const object& other) : m_ptr(other.m_ptr) {
  Py_XINCREF(m_ptr);
}

object& object::operator=(const object& other) {
  Py_XDECREF(m_ptr);
  m_ptr = other.m_ptr;
  Py_XINCREF(m_ptr);
  return *this;
}

object::object(object&& other) : m_ptr(other.m_ptr) {
  other.m_ptr = nullptr;
}

object& object::operator=(object&& other) {
  Py_XDECREF(m_ptr);
  m_ptr = other.m_ptr;
  other.m_ptr = nullptr;
  return *this;
}

object::~object() {
  Py_XDECREF(m_ptr);
}

} // namespace python

python_reader::python_reader(std::string script,
                             std::string sample_function,
                             std::string num_samples_function,
                             std::string sample_dims_function)
  : generic_data_reader(true) {
  int status;

  // Execute Python script
  auto& manager = python::manager::get_instance();
  const auto lock = manager.get_mutex_guard();
  auto&& f = std::fopen(script.c_str(), "r");
  if (f == nullptr) {
    LBANN_ERROR("failed to open file (" + script + ")");
  }
  status = PyRun_SimpleFile(f, script.c_str());
  if (status) {
    manager.check_error(status);
  }
  status = std::fclose(f);
  if (status) {
    LBANN_ERROR("failed to close file (" + script + ")");
  }
  python::object module = PyImport_ImportModule("__main__");

  // Get number of samples
  python::object num_func
    = PyObject_GetAttrString(module, num_samples_function.c_str());
  python::object num = PyObject_CallObject(num_func, nullptr);
  m_num_samples = PyLong_AsLong(num);
  manager.check_error();

  // Get sample dimensions
  python::object dims_func
    = PyObject_GetAttrString(module, sample_dims_function.c_str());
  python::object dims = PyObject_CallObject(dims_func, nullptr);
  dims = PyObject_GetIter(dims);
  for (auto d = PyIter_Next(dims); d != nullptr; d = PyIter_Next(dims)) {
    m_sample_dims.push_back(PyLong_AsLong(d));
    Py_DECREF(d);
  }
  manager.check_error();

  // Get sample function
  m_sample_function
    = PyObject_GetAttrString(module, sample_function.c_str());

}

const std::vector<int> python_reader::get_data_dims() const {
  std::vector<int> dims;
  for (const auto& d : m_sample_dims) {
    dims.push_back(d);
  }
  return dims;
}
int python_reader::get_num_labels() const {
  return 1;
}
int python_reader::get_linearized_data_size() const {
  const auto& dims = get_data_dims();
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<int>());
}
int python_reader::get_linearized_label_size() const {
  return get_num_labels();
}

bool python_reader::fetch_datum(CPUMat& X, int data_id, int col) {

  // Lock mutex for the scope of this function
  auto& manager = python::manager::get_instance();
  const auto lock = manager.get_mutex_guard();

  // Get sample with Python
  python::object args = Py_BuildValue("(i)", data_id);
  python::object sample = PyObject_CallObject(m_sample_function, args);
  sample = PyObject_GetIter(sample);

  // Extract sample entries from Python iterator
  const El::Int sample_size = get_linearized_data_size();
  for (El::Int row = 0; row < sample_size; ++row) {
    python::object val = PyIter_Next(sample);
    X(row, col) = PyFloat_AsDouble(val);
  }
  if (PyErr_Occurred()) { LBANN_ERROR("Python error detected"); }

  return true;
}

bool python_reader::fetch_label(CPUMat& Y, int data_id, int col) {
  return true;
}

void python_reader::load() {
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

} // namespace lbann

#endif // LBANN_HAS_PYTHON
