#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <omp.h>
#include <random>
#include <numeric>
#include "../lssg/utils.hh"
#include "../lssg/visit_list.hh"
#include "../lssg/space_dist.hh"

namespace benchmark {
auto fvecs_read(const std::string &filename, size_t &d_out, size_t &n_out) -> float *
{
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + filename);
  }
  int d;
  in.read(reinterpret_cast<char *>(&d), 4);
  d_out = d;
  // calculate file size
  in.seekg(0, std::ios::beg);
  in.seekg(0, std::ios::end);
  size_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  size_t n    = file_size / (4 + d * sizeof(float));
  n_out       = n;
  float *data = new float[d * n];
  for (size_t i = 0; i < n; ++i) {
    in.read(reinterpret_cast<char *>(&d), 4);
    in.read(reinterpret_cast<char *>(data + i * d), d * sizeof(float));
  }
  in.close();
  std::cout << "Loaded fvecs: " << n << " vectors of dimension " << d << " from " << filename << std::endl;
  return data;
}

auto LoadRange(const std::string &location) -> std::vector<lssg::lssg_range<int>>
{
  std::vector<lssg::lssg_range<int>> query_filters;
  std::ifstream                       ifs(location, std::ios::binary);
  if (!ifs.is_open()) {
    std::cout << "Fail to open: " << location << std::endl;
    std::abort();
  }
  // check meta size
  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  auto n_query = file_size / (2 * sizeof(int));
  query_filters.resize(n_query);
  for (size_t i = 0; i < n_query; ++i) {
    int l, u;
    ifs.read(reinterpret_cast<char *>(&l), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&u), sizeof(int));
    query_filters[i] = lssg::lssg_range<int>{l, u};
  }
  ifs.close();
  // LOG first 10 query filters
  std::string qf_str;
  for (size_t i = 0; i < std::min<size_t>(10, n_query); ++i) {
    qf_str += "[" + std::to_string(query_filters[i].l_) + "," + std::to_string(query_filters[i].u_) + "]";
  }
  std::cout << "First 10 query filters: " << qf_str << std::endl;
  return query_filters;
}

template <typename att_t, typename filter_t>
auto GenGT(size_t nb, size_t nq, size_t d, size_t k, const std::vector<filter_t> &filter, const float *basevec,
    const float *queryvec, std::vector<att_t> attvec, const std::string &space)
    -> std::vector<std::vector<lssg::label_t>>
{
  lssg::SpaceInterface<float> *space_ptr;
  if (space == "l2") {
    space_ptr = new lssg::L2Space(d);
  } else if (space == "ip") {
    space_ptr = new lssg::InnerProductSpace(d);
  } else {
    throw std::runtime_error("unsupported space type " + space + ", supported: l2, ip");
  }
  auto                                      fstdistfunc     = space_ptr->get_dist_func();
  auto                                      dist_func_param = space_ptr->get_dist_func_param();
  std::vector<std::vector<lssg::label_t>> gt(nq);
  size_t                                    iq;
  std::cout << "Generating ground truth..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic) \
    shared(gt, queryvec, basevec, attvec, fstdistfunc, dist_func_param, k)
  for (iq = 0; iq < nq; ++iq) {
    std::vector<lssg::dist_id_pair> gt_cand;
    for (int ib = 0; ib < nb; ++ib) {
      if (filter[iq].Test(attvec[ib])) {
        auto dist = fstdistfunc(queryvec + iq * d, basevec + ib * d, dist_func_param);
        PUSH_HEAP(gt_cand, dist, ib);
        if (gt_cand.size() > k) {
          POP_HEAP(gt_cand);
        }
      }
    }

    gt[iq].resize(gt_cand.size());
    for (size_t i = 0; i < gt_cand.size(); ++i) {
      gt[iq][i] = gt_cand[i].id_;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Ground truth generation took " << elapsed.count() << " seconds for " << nq << " queries, QPS: " << (float)nq / elapsed.count()
            << std::endl;
  delete space_ptr;
  return gt;
}

auto LoadBitmap(const std::string &bitmap_file, size_t n) -> std::vector<lssg::lssg_bitset<lssg::label_t>>
{
  std::ifstream in(bitmap_file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + bitmap_file);
  }
  std::vector<lssg::lssg_bitset<lssg::label_t>> all_bitmap;
  while (!in.eof()) {
    int k;
    in.read(reinterpret_cast<char *>(&k), sizeof(int));
    if (in.eof()) {
      break;
    }
    lssg::lssg_bitset<lssg::label_t> bitmap(n);
    for (int i = 0; i < k; ++i) {
      unsigned int ib;
      in.read(reinterpret_cast<char *>(&ib), sizeof(unsigned int));
      if (ib >= n) {
        throw std::runtime_error("bitmap index out of range: " + std::to_string(ib) + ", n: " + std::to_string(n));
      }
      bitmap.Set(ib);
    }
    all_bitmap.emplace_back(std::move(bitmap));
  }
  std::cout << "Loaded bitmap: " << all_bitmap.size() << std::endl;
  return all_bitmap;
}

auto GenBitmap(size_t npass, size_t nb) -> lssg::lssg_bitset<lssg::label_t>
{
  // npass is the number of 1 and others are 0 in 0--nb-1
  lssg::lssg_bitset<lssg::label_t> bitmap(nb);
  bitmap.Clear();
  std::vector<lssg::label_t> idx(nb);
  std::iota(idx.begin(), idx.end(), 0);
  std::random_device rd;
  std::mt19937       g(rd());
  std::shuffle(idx.begin(), idx.end(), g);
  for (size_t i = 0; i < npass; ++i) {
    bitmap.Set(idx[i]);
  }
  return std::move(bitmap);
}

auto LoadGroundTruth(const std::string &gt_file) -> std::vector<std::vector<lssg::label_t>>
{
  std::ifstream in(gt_file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + gt_file);
  }
  std::vector<std::vector<lssg::label_t>> all_gt;
  while (!in.eof()) {
    int k;
    in.read(reinterpret_cast<char *>(&k), sizeof(int));
    if (in.eof()) {
      break;
    }
    std::vector<lssg::label_t> gt(k);
    for (int i = 0; i < k; ++i) {
      unsigned int ib;
      in.read(reinterpret_cast<char *>(&ib), sizeof(unsigned int));
      gt[i] = ib;
    }
    all_gt.emplace_back(gt);
  }
  // example
  std::cout << "Loaded ground truth: " << all_gt.size() << std::endl;
  std::cout << "Example: first query has " << all_gt[0].size() << std::endl;
  for (auto ib : all_gt[0]) {
    std::cout << ib << ",";
  }
  std::cout << std::endl;
  return all_gt;
}

template <typename att_t>
auto LoadAttVec(const std::string att_file) -> std::vector<att_t>
{
  std::ifstream in(att_file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + att_file);
  }
  in.seekg(0, std::ios::end);
  size_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  size_t n = file_size / sizeof(att_t);
  if (file_size % sizeof(att_t) != 0) {
    throw std::runtime_error("File size is not a multiple of att_t");
  }
  std::vector<att_t> att_vec(n);
  for (size_t i = 0; i < n; ++i) {
    in.read(reinterpret_cast<char *>(&att_vec[i]), sizeof(att_t));
  }
  in.close();
  std::cout << "Loaded att_vec: " << att_file << ", size: " << n << std::endl;
  return att_vec;
}

auto CalculateRecall(
    const std::vector<std::vector<lssg::label_t>> &gt, const std::vector<std::vector<lssg::label_t>> &res) -> float
{
  size_t n       = std::min(gt.size(), res.size());
  size_t total   = 0;
  size_t correct = 0;
  for (size_t i = 0; i < n; ++i) {
    total += gt[i].size();
    for (auto ib : res[i]) {
      if (std::find(gt[i].begin(), gt[i].end(), ib) != gt[i].end()) {
        correct++;
      }
    }
  }
  return static_cast<float>(correct) / total;
}

auto CalculateRecall(const std::vector<lssg::label_t> &gt, const std::vector<lssg::label_t> &res) -> float
{
  size_t total   = gt.size();
  size_t correct = 0;
  for (auto ib : res) {
    if (std::find(gt.begin(), gt.end(), ib) != gt.end()) {
      correct++;
    }
  }
  return static_cast<float>(correct) / total;
}

auto LoadLabelSets(const std::string &filename) -> std::vector<lssg::labelset_t>
{
  std::ifstream in(filename);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + filename);
  }
  std::vector<lssg::labelset_t> label_sets;
  std::string                     line;

  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    lssg::labelset_t labels;
    std::stringstream  ss(line);
    std::string        token;

    while (std::getline(ss, token, ',')) {
      token.erase(0, token.find_first_not_of(" \t\n\r\f\v"));
      token.erase(token.find_last_not_of(" \t\n\r\f\v") + 1);

      if (!token.empty()) {
        try {
          int val = std::stoi(token);
          labels.emplace_back(static_cast<lssg::labelsetmember_t>(val));
        } catch (const std::exception &e) {
          std::cerr << "Warning: Failed to parse token '" << token << "' in line: " << line << std::endl;
          assert(0);
          exit(1);
        }
      }
    }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

    if (!labels.empty()) {
      label_sets.emplace_back(std::move(labels));
    }
  }

  std::cout << "Loaded label sets: " << label_sets.size() << std::endl;
  return label_sets;
}
inline bool LabelFilterMatch(
    const lssg::labelset_t &base_set, const lssg::labelset_t &query_set, lssg::LabelFilterConfig::Type query_type)
{
  switch (query_type) {
    case lssg::LabelFilterConfig::LABEL_CONTAINMENT: {
      // Two-pointer approach: check if base_set contains all elements of query_set
      // Both sets are sorted
      size_t query_idx = 0;
      size_t base_idx  = 0;

      while (query_idx < query_set.size() && base_idx < base_set.size()) {
        if (query_set[query_idx] == base_set[base_idx]) {
          // Found this query label in base, move to next query label
          query_idx++;
          base_idx++;
        } else if (query_set[query_idx] < base_set[base_idx]) {
          // Query label not in base (base advanced past it)
          return false;
        } else {
          // base_set[base_idx] < query_set[query_idx]
          // Keep advancing base pointer to find query label
          base_idx++;
        }
      }

      // If we matched all query labels, return true
      return query_idx == query_set.size();
    }
    case lssg::LabelFilterConfig::LABEL_EQUALITY: return base_set == query_set;
    case lssg::LabelFilterConfig::LABEL_OVERLAP: {
      size_t i = 0, j = 0;
      while (i < base_set.size() && j < query_set.size()) {
        if (base_set[i] == query_set[j]) {
          return true;
        }
        if (base_set[i] < query_set[j]) {
          ++i;
        } else {
          ++j;
        }
      }
      return false;
    }
    default: throw std::runtime_error("Unknown label filter type");
  }
}

auto GenLabelSetGT(size_t nb, size_t nq, size_t d, size_t k, const float *basevec, const float *queryvec,
    const std::vector<lssg::labelset_t> &base_sets, const std::vector<lssg::labelset_t> &query_sets,
    lssg::LabelFilterConfig::Type query_type, const std::string &space) -> std::vector<std::vector<lssg::label_t>>
{
  if (base_sets.size() != nb) {
    throw std::runtime_error("base_sets size does not match nb");
  }
  if (query_sets.size() != nq) {
    throw std::runtime_error("query_sets size does not match nq");
  }

  lssg::SpaceInterface<float> *space_ptr;
  if (space == "l2") {
    space_ptr = new lssg::L2Space(d);
  } else if (space == "ip") {
    space_ptr = new lssg::InnerProductSpace(d);
  } else {
    throw std::runtime_error("unsupported space type " + space + ", supported: l2, ip");
  }

  auto                                      fstdistfunc     = space_ptr->get_dist_func();
  auto                                      dist_func_param = space_ptr->get_dist_func_param();
  std::vector<std::vector<lssg::label_t>> gt(nq);

  std::cout << "Generating label ground truth for type " << query_type << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic)
  for (size_t iq = 0; iq < nq; ++iq) {
    std::vector<lssg::dist_id_pair> gt_cand;
    gt_cand.reserve(k + 4);
    for (size_t ib = 0; ib < nb; ++ib) {
      if (!LabelFilterMatch(base_sets[ib], query_sets[iq], query_type)) {
        continue;
      }
      auto dist = fstdistfunc(queryvec + iq * d, basevec + ib * d, dist_func_param);
      PUSH_HEAP(gt_cand, dist, ib);
      if (gt_cand.size() > k) {
        POP_HEAP(gt_cand);
      }
    }

    gt[iq].resize(gt_cand.size());
    for (size_t i = 0; i < gt_cand.size(); ++i) {
      gt[iq][i] = gt_cand[i].id_;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Ground truth generation took " << elapsed.count() << " seconds for " << nq << " queries, QPS: " << (float)nq / elapsed.count()
            << std::endl;

  delete space_ptr;
  return gt;
}

}  // namespace benchmark