#include "bench_utils.hh"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

lssg::LabelFilterConfig::Type ParseLabelQueryType(const std::string &type_str)
{
  if (type_str == "containment") {
    return lssg::LabelFilterConfig::LABEL_CONTAINMENT;
  }
  if (type_str == "equality") {
    return lssg::LabelFilterConfig::LABEL_EQUALITY;
  }
  if (type_str == "overlap") {
    return lssg::LabelFilterConfig::LABEL_OVERLAP;
  }
  throw std::runtime_error("Unsupported label query type: " + type_str);
}

int main(int argc, char **argv)
{
  std::string base_vec_file, query_vec_file;
  std::string base_label_file, query_label_file;
  std::string gt_file, space, type_str;
  size_t      k = 10;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--basevec") == 0) {
      base_vec_file = argv[++i];
    } else if (strcmp(argv[i], "--queryvec") == 0) {
      query_vec_file = argv[++i];
    } else if (strcmp(argv[i], "--base_labelset") == 0) {
      base_label_file = argv[++i];
    } else if (strcmp(argv[i], "--query_labelset") == 0) {
      query_label_file = argv[++i];
    } else if (strcmp(argv[i], "--gt_file") == 0) {
      gt_file = argv[++i];
    } else if (strcmp(argv[i], "--space") == 0) {
      space = argv[++i];
    } else if (strcmp(argv[i], "--type") == 0) {
      type_str = argv[++i];
    } else if (strcmp(argv[i], "--k") == 0) {
      k = std::stoul(argv[++i]);
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(argv[i]));
    }
  }

  if (base_vec_file.empty() || query_vec_file.empty() || base_label_file.empty() || query_label_file.empty() ||
      gt_file.empty() || space.empty() || type_str.empty()) {
    throw std::runtime_error("Missing required arguments for gen_label_gt");
  }

  auto query_type = ParseLabelQueryType(type_str);

  size_t base_dim = 0, nb = 0;
  auto   base_vecs = benchmark::fvecs_read(base_vec_file, base_dim, nb);

  size_t query_dim = 0, nq = 0;
  auto   query_vecs = benchmark::fvecs_read(query_vec_file, query_dim, nq);

  if (base_dim != query_dim) {
    delete[] base_vecs;
    delete[] query_vecs;
    throw std::runtime_error("Base and query vector dimensions do not match");
  }

  auto base_labelsets  = benchmark::LoadLabelSets(base_label_file);
  auto query_labelsets = benchmark::LoadLabelSets(query_label_file);

  if (base_labelsets.size() != nb) {
    delete[] base_vecs;
    delete[] query_vecs;
    throw std::runtime_error("Base label set count does not match base vector count");
  }
  if (query_labelsets.size() != nq) {
    nq = std::min(query_labelsets.size(), nq);
    query_labelsets.resize(nq);
  }

  auto gt = benchmark::GenLabelSetGT(nb, nq, base_dim, k, base_vecs, query_vecs, base_labelsets, query_labelsets,
      query_type, space);

  std::ofstream ofs(gt_file, std::ios::binary);
  if (!ofs.is_open()) {
    delete[] base_vecs;
    delete[] query_vecs;
    throw std::runtime_error("Cannot open gt file for writing: " + gt_file);
  }

  for (size_t iq = 0; iq < gt.size(); ++iq) {
    int count = static_cast<int>(gt[iq].size());
    ofs.write(reinterpret_cast<const char *>(&count), sizeof(int));
    for (const auto &id : gt[iq]) {
      unsigned int uid = static_cast<unsigned int>(id);
      ofs.write(reinterpret_cast<const char *>(&uid), sizeof(unsigned int));
    }
  }
  ofs.close();

  std::cout << "Ground truth generated: " << gt_file << std::endl;

  delete[] base_vecs;
  delete[] query_vecs;
  return 0;
}
