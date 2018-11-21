#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <cstdlib> // exit
#include "mem.hpp"
#include "lbann/data_readers/offline_patches_npz.hpp"

using namespace lbann;

void print_all_samples(const offline_patches_npz& data) {
  const size_t num_samples = data.get_num_samples();
  for(size_t i=0u; i < num_samples; ++i) {
    offline_patches_npz::sample_t sample = data.get_sample(i);
#if 0
    std::cout << "(['" << sample.first[0] << "', '"
                       << sample.first[1] << "', '"
                       << sample.first[2] << "'], '"
              << static_cast<unsigned int>(sample.second) << "')" << std::endl;
#else
    std::cout << sample.first[0] << ' '
              << sample.first[1] << ' '
              << sample.first[2] << ' '
              << static_cast<unsigned int>(sample.second) << std::endl;
#endif
  }
}

void load(const std::string& file_name, offline_patches_npz& data, const int out_mode, const size_t n_first = 0u) {
  const bool keep_file_lists = ((out_mode == 4)? true : false);

  if (!data.load(file_name, n_first, keep_file_lists)) {
    std::cerr << "Failed to load " << file_name << std::endl;
    exit(0);
  }
}


int main(int argc, char** argv)
{
  if ((argc < 4) || (argc > 7)) {
    std::cout << "Uasge: > " << argv[0] << " npz_file in_mode out_mode [arg1 [arg2 [out_file]]]" << std::endl;
    std::cout << "         in_mode 0: load all data" << std::endl;
    std::cout << "         in_mode 1: load first n(arg1) samples" << std::endl;
    std::cout << "         in_mode 2: load all data and proceed to out_mode 2" << std::endl;
    std::cout << "        out_mode 0: show data description" << std::endl;
    std::cout << "        out_mode 1: print the list of samples to stdout" << std::endl;
    std::cout << "        out_mode 2: print the number of samples in first n(arg1) sub directories" << std::endl;
    std::cout << "        out_mode 3: print the subdirectory names of samples to stdout" << std::endl;
    std::cout << "        out_mode 4: write samples selected by the id range, between id_start(arg1) and id_end(arg2)" << std::endl;
    std::cout << "                    The chosen samples are written to out_file" << std::endl;
    std::cout << "                    id_start is inclusive, and id_end is exclusive." << std::endl;
    return 0;
  }

  std::string file_name(argv[1]);
  int in_mode = atoi(argv[2]);
  int out_mode = atoi(argv[3]);
  size_t num_subdirs = 0u;
  size_t n_first = 0u;

  offline_patches_npz data;


  switch (in_mode) {
    case 0: { // load all data
      load(file_name, data, out_mode);
    } break;
    case 1: { // load first_n samples
      n_first = static_cast<size_t>(atoi(argv[4]));
      load(file_name, data, out_mode, n_first);
    } break;
    case 2: {
      if (out_mode != 2) {
        std::cout << "Changing out_mode to 2, to count the number of samples in first "
                  << num_subdirs << " directories" << std::endl;
        out_mode = 2;
      }
      load(file_name, data, out_mode);
      if (argc != 5) {
        std::cout << "The number of subdir argument (arg1)  is missing" << std::endl;
        exit(0);
      }
      num_subdirs = static_cast<size_t>(atoi(argv[4]));
    } break;
    default:
      std::cout << "Invalid in_mode: " << in_mode << std::endl;
      return 0;
  }

  switch (out_mode) {
    case 0: { // show data description
      print_mem("Memory status :");
      std::cout << data.get_description() << std::endl;
    } break;
    case 1: { // print the list of samples to stdout
      print_all_samples(data);
    } break;
    case 2: { // print the number of samples in first n sub directories
      if (in_mode != 2) {
        std::cout << "in_mode was not 2 but " << in_mode << std::endl;
      } else {
        std::cout << "Number of subdirs: " << num_subdirs << std::endl;
        std::cout << "Number of samples: " << data.count_samples(num_subdirs) << std::endl;
      }
    } break;
    case 3: { // print the subdirectory names of samples
      const std::vector<std::string> root_names = data.get_file_roots();
      for(auto&& r: root_names) {
        std::cout << r << std::endl;
      }
    } break;
    case 4: { // write selected samples into a new file
      if (argc != 7) {
        std::cout << "Requires the arguments: id_start, id_end, and out_file." << std::endl;
        exit(0);
      }
      size_t id_start = static_cast<size_t>(atoi(argv[4]));
      size_t id_end= static_cast<size_t>(atoi(argv[5]));
      std::string out_file(argv[6]);

      if (out_file == file_name) {
        std::cout << "Cannot overwrite the data file" << std::endl;
        exit(0);
      }
      bool ok = data.select(out_file, id_start, id_end);
      if (!ok) {
        std::cout << "Failed to select [" << id_start << ", " << id_end << std::endl;
        exit(0);
      }
    } break;
    default:
      std::cout << "Invalid out_mode: " << in_mode << std::endl;
      return 0;
  }

  return 0;
}
