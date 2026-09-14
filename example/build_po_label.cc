#include "../lssg/poindex.hh"
#include "bench_utils.hh"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

int main(int argc, char **argv)
{
  size_t      m = 0, efc = 0;
  std::string basevec, baseset, space, index_location, scope_location;
  int         t = omp_get_max_threads();
  size_t      o = 4, wp = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--m") == 0) {
      m = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--efc") == 0) {
      efc = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--basevec") == 0) {
      basevec = argv[++i];
    } else if (strcmp(argv[i], "--labelset") == 0) {
      baseset = argv[++i];
    } else if (strcmp(argv[i], "--space") == 0) {
      space = argv[++i];
    } else if (strcmp(argv[i], "--threads") == 0) {
      t = std::stoi(argv[++i]);
    } else if (strcmp(argv[i], "--index_location") == 0) {
      index_location = argv[++i];
    } else if (strcmp(argv[i], "--scope_location") == 0) {
      scope_location = argv[++i];
    } else if (strcmp(argv[i], "--o") == 0) {
      o = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--wp") == 0) {
      wp = std::stoul(argv[++i]);
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argv[i]));
    }
  }

  if (m == 0 || efc == 0 || basevec.empty() || baseset.empty() || space.empty() || index_location.empty() ||
      scope_location.empty()) {
    throw std::runtime_error("Missing required arguments for build_po_label");
  }

  size_t dim = 0, maxN = 0;
  float *basevecs = benchmark::fvecs_read(basevec, dim, maxN);
  auto   label_sets = benchmark::LoadLabelSets(baseset);
  if (label_sets.size() != maxN) {
    delete[] basevecs;
    throw std::runtime_error("Base vector count and label set count do not match");
  }

  size_t max_layer = 8;

  auto label_kernel = std::make_unique<lssg::LabelSetScopeKernel>(max_layer, maxN, baseset);
  auto init_start = std::chrono::high_resolution_clock::now();
  label_kernel->Init();
  auto init_end = std::chrono::high_resolution_clock::now();
  std::cout << "Scope kernel initialized in " << std::chrono::duration<double>(init_end - init_start).count()
            << " seconds" << std::endl;
  lssg::PoIndex<lssg::labelset_t, float, lssg::LabelSetScopeKernel> index(maxN, dim, m, efc, space,
      label_kernel);

  std::vector<int> ids(maxN);
  std::iota(ids.begin(), ids.end(), 0); 

  auto start = std::chrono::high_resolution_clock::now();
  size_t i;
#pragma omp parallel for num_threads(t) schedule(dynamic) shared(index, i)
  for (i = 0; i < maxN; ++i) {
    auto cur_id = ids[i];
    index.insert(cur_id, basevecs + cur_id * dim, label_sets[cur_id]);
    if(i % 10000 == 0){
      std::cout << "inserting "<< i << " / " << maxN << std::endl;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Index built in " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;

  index.save(index_location, scope_location);
  std::cout << "Index saved to: " << index_location << std::endl;

  delete[] basevecs;
  return 0;
}
