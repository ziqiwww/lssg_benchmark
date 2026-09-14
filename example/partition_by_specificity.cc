#include "bench_utils.hh"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Specificity bins
enum SpecificityBin
{
  BIN_0_20 = 0,
  BIN_20_40,
  BIN_40_60,
  BIN_60_80,
  BIN_80_100,
  NUM_BINS
};

const char *BinNames[NUM_BINS] = {"0-20", "20-40", "40-60", "60-80", "80-100"};

SpecificityBin GetSpecificityBin(float specificity_pct)
{
  if (specificity_pct < 20.0f)
    return BIN_0_20;
  if (specificity_pct < 40.0f)
    return BIN_20_40;
  if (specificity_pct < 60.0f)
    return BIN_40_60;
  if (specificity_pct < 80.0f)
    return BIN_60_80;
  return BIN_80_100;
}

struct QueryInfo
{
  size_t          original_index;
  float           specificity;
  size_t          passing_count;
  SpecificityBin  bin;
};

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

void WriteFvecs(const std::string &filename, const std::vector<std::vector<float>> &vecs, size_t dim)
{
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  for (const auto &vec : vecs) {
    if (vec.size() != dim) {
      throw std::runtime_error("Vector dimension mismatch");
    }
    int d = static_cast<int>(dim);
    ofs.write(reinterpret_cast<const char *>(&d), sizeof(int));
    ofs.write(reinterpret_cast<const char *>(vec.data()), dim * sizeof(float));
  }

  ofs.close();
  std::cout << "Wrote " << vecs.size() << " vectors to " << filename << std::endl;
}

void WriteBinVecs(const std::string &filename, const std::vector<std::vector<float>> &vecs, size_t dim)
{
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  // Write header: number of vectors (n) and dimension (d)
  int n = static_cast<int>(vecs.size());
  int d = static_cast<int>(dim);
  ofs.write(reinterpret_cast<const char *>(&n), sizeof(int));
  ofs.write(reinterpret_cast<const char *>(&d), sizeof(int));

  // Write all vectors (n * d * 4 bytes)
  for (const auto &vec : vecs) {
    if (vec.size() != dim) {
      throw std::runtime_error("Vector dimension mismatch");
    }
    ofs.write(reinterpret_cast<const char *>(vec.data()), dim * sizeof(float));
  }

  ofs.close();
  std::cout << "Wrote " << vecs.size() << " vectors to " << filename << " (bin format)" << std::endl;
}

void WriteLabelSets(const std::string &filename, const std::vector<lssg::labelset_t> &labelsets)
{
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  for (const auto &labels : labelsets) {
    if (labels.empty()) {
      ofs << "\n";
    } else {
      for (size_t i = 0; i < labels.size(); ++i) {
        if (i > 0)
          ofs << ",";
        ofs << labels[i];
      }
      ofs << "\n";
    }
  }

  ofs.close();
  std::cout << "Wrote " << labelsets.size() << " label sets to " << filename << std::endl;
}

void WriteStats(const std::string &filename, const std::string &dataset_name, const std::string &scenario,
    SpecificityBin bin, const std::vector<QueryInfo> &queries, size_t total_base_items)
{
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  // Compute statistics
  float min_spec = 100.0f, max_spec = 0.0f, sum_spec = 0.0f;
  size_t min_pass = SIZE_MAX, max_pass = 0, sum_pass = 0;

  for (const auto &q : queries) {
    min_spec = std::min(min_spec, q.specificity);
    max_spec = std::max(max_spec, q.specificity);
    sum_spec += q.specificity;

    min_pass = std::min(min_pass, q.passing_count);
    max_pass = std::max(max_pass, q.passing_count);
    sum_pass += q.passing_count;
  }

  float avg_spec = queries.empty() ? 0.0f : sum_spec / queries.size();
  float avg_pass = queries.empty() ? 0.0f : static_cast<float>(sum_pass) / queries.size();

  ofs << "Bin: " << BinNames[bin] << "%\n";
  ofs << "Scenario: " << scenario << "\n";
  ofs << "Dataset: " << dataset_name << "\n";
  ofs << "Number of queries: " << queries.size() << "\n";
  ofs << "Total base data items: " << total_base_items << "\n";
  ofs << "\nSpecificity statistics:\n";
  ofs << "  Min specificity: " << min_spec << "%\n";
  ofs << "  Max specificity: " << max_spec << "%\n";
  ofs << "  Mean specificity: " << avg_spec << "%\n";
  ofs << "\nPassing items statistics:\n";
  ofs << "  Min passing items: " << min_pass << "\n";
  ofs << "  Max passing items: " << max_pass << "\n";
  ofs << "  Mean passing items: " << avg_pass << "\n";
  ofs << "\nQuery indices: ";

  size_t max_print = std::min<size_t>(100, queries.size());
  for (size_t i = 0; i < max_print; ++i) {
    if (i > 0)
      ofs << ",";
    ofs << queries[i].original_index;
  }
  if (queries.size() > max_print) {
    ofs << "... and " << (queries.size() - max_print) << " more";
  }
  ofs << "\n";

  ofs.close();
}

int main(int argc, char **argv)
{
  std::string base_vec_file, query_vec_file;
  std::string base_label_file, query_label_file;
  std::string output_dir, dataset_name, scenario_str;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--base_vec") == 0) {
      base_vec_file = argv[++i];
    } else if (strcmp(argv[i], "--query_vec") == 0) {
      query_vec_file = argv[++i];
    } else if (strcmp(argv[i], "--base_labelset") == 0) {
      base_label_file = argv[++i];
    } else if (strcmp(argv[i], "--query_labelset") == 0) {
      query_label_file = argv[++i];
    } else if (strcmp(argv[i], "--output_dir") == 0) {
      output_dir = argv[++i];
    } else if (strcmp(argv[i], "--dataset_name") == 0) {
      dataset_name = argv[++i];
    } else if (strcmp(argv[i], "--scenario") == 0) {
      scenario_str = argv[++i];
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(argv[i]));
    }
  }

  if (base_vec_file.empty() || query_vec_file.empty() || base_label_file.empty() || query_label_file.empty() ||
      output_dir.empty() || dataset_name.empty() || scenario_str.empty()) {
    std::cerr << "Usage: " << argv[0] << " --base_vec <file> --query_vec <file> "
              << "--base_labelset <file> --query_labelset <file> "
              << "--output_dir <dir> --dataset_name <name> --scenario <containment|equality|overlap>\n";
    return 1;
  }

  auto scenario = ParseLabelQueryType(scenario_str);

  // Load data
  std::cout << "================================================================================" << std::endl;
  std::cout << "Loading data..." << std::endl;
  std::cout << "================================================================================" << std::endl;

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
    std::cout << "Warning: Query labelset count (" << query_labelsets.size() << ") != query vector count (" << nq
              << ")" << std::endl;
    nq = std::min(query_labelsets.size(), nq);
    query_labelsets.resize(nq);
  }

  std::cout << "Dataset: " << dataset_name << std::endl;
  std::cout << "Scenario: " << scenario_str << std::endl;
  std::cout << "Base items: " << nb << std::endl;
  std::cout << "Query items: " << nq << std::endl;
  std::cout << "Vector dimension: " << base_dim << std::endl;

  // Compute specificity for each query
  std::cout << "\n================================================================================" << std::endl;
  std::cout << "Computing specificity for " << nq << " queries..." << std::endl;
  std::cout << "================================================================================" << std::endl;

  std::vector<QueryInfo> query_infos(nq);

#pragma omp parallel for schedule(dynamic, 100)
  for (size_t iq = 0; iq < nq; ++iq) {
    size_t passing_count = 0;

    for (size_t ib = 0; ib < nb; ++ib) {
      if (benchmark::LabelFilterMatch(base_labelsets[ib], query_labelsets[iq], scenario)) {
        passing_count++;
      }
    }

    float specificity = (static_cast<float>(passing_count) / nb) * 100.0f;

    query_infos[iq].original_index = iq;
    query_infos[iq].specificity    = specificity;
    query_infos[iq].passing_count  = passing_count;
    query_infos[iq].bin            = GetSpecificityBin(specificity);

    if ((iq + 1) % 1000 == 0) {
#pragma omp critical
      {
        std::cout << "  Progress: " << (iq + 1) << "/" << nq << " (" << (100.0 * (iq + 1) / nq) << "%)" << std::endl;
      }
    }
  }

  std::cout << "  Completed: " << nq << "/" << nq << " (100.0%)" << std::endl;

  // Calculate overall average specificity
  float total_specificity = 0.0f;
  for (const auto &qinfo : query_infos) {
    total_specificity += qinfo.specificity;
  }
  float avg_specificity = total_specificity / nq;
  
  std::cout << "\nOverall average specificity: " << avg_specificity << "%" << std::endl;

  // Partition by bins
  std::map<SpecificityBin, std::vector<QueryInfo>> bins;
  for (const auto &qinfo : query_infos) {
    bins[qinfo.bin].push_back(qinfo);
  }

  // Print statistics
  std::cout << "\nSpecificity distribution:" << std::endl;
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
  printf("%-12s %-10s %-12s %-15s %s\n", "Bin", "Count", "Percentage", "Avg Specificity", "Avg Pass Items");
  std::cout << "--------------------------------------------------------------------------------" << std::endl;

  for (int b = 0; b < NUM_BINS; ++b) {
    auto bin         = static_cast<SpecificityBin>(b);
    auto count       = bins[bin].size();
    auto percentage  = (static_cast<float>(count) / nq) * 100.0f;
    float avg_passing = 0.0f;
    float avg_spec_bin = 0.0f;

    if (!bins[bin].empty()) {
      size_t sum = 0;
      float sum_spec = 0.0f;
      for (const auto &q : bins[bin]) {
        sum += q.passing_count;
        sum_spec += q.specificity;
      }
      avg_passing = static_cast<float>(sum) / bins[bin].size();
      avg_spec_bin = sum_spec / bins[bin].size();
    }

    printf("%-12s %-10zu %6.2f%%      %10.2f%%      %10.0f / %zu\n", BinNames[b], count, percentage, avg_spec_bin, avg_passing, nb);
  }

  std::cout << "\nTotal queries: " << nq << std::endl;
  std::cout << "Total base data items: " << nb << std::endl;

  // Write partitioned files
  std::cout << "\n================================================================================" << std::endl;
  std::cout << "Writing partitioned files to " << output_dir << "..." << std::endl;
  std::cout << "================================================================================" << std::endl;

  // Create output directory
  std::string mkdir_cmd = "mkdir -p " + output_dir;
  system(mkdir_cmd.c_str());

  for (int b = 0; b < NUM_BINS; ++b) {
    auto bin = static_cast<SpecificityBin>(b);

    if (bins[bin].empty()) {
      std::cout << "  Skipping bin " << BinNames[b] << " (empty)" << std::endl;
      continue;
    }

    // Extract vectors and labelsets for this bin
    std::vector<std::vector<float>> bin_vecs;
    std::vector<lssg::labelset_t> bin_labelsets;

    for (const auto &qinfo : bins[bin]) {
      size_t idx = qinfo.original_index;

      // Extract vector
      std::vector<float> vec(base_dim);
      for (size_t d = 0; d < base_dim; ++d) {
        vec[d] = query_vecs[idx * base_dim + d];
      }
      bin_vecs.push_back(vec);

      // Extract labelset
      bin_labelsets.push_back(query_labelsets[idx]);
    }

    // Generate output filenames
    std::string vec_filename = output_dir + "/" + dataset_name + "_" + scenario_str + "_specificity" + BinNames[b] +
                               "_query.fvecs";
    std::string bin_filename = output_dir + "/" + dataset_name + "_" + scenario_str + "_specificity" + BinNames[b] +
                               "_query.bin";
    std::string labelset_filename = output_dir + "/" + dataset_name + "_" + scenario_str + "_specificity" +
                                    BinNames[b] + "_query.txt";
    std::string stats_filename = output_dir + "/" + dataset_name + "_" + scenario_str + "_specificity" + BinNames[b] +
                                 "_stats.txt";

    // Write files
    WriteFvecs(vec_filename, bin_vecs, base_dim);
    WriteBinVecs(bin_filename, bin_vecs, base_dim);
    WriteLabelSets(labelset_filename, bin_labelsets);
    WriteStats(stats_filename, dataset_name, scenario_str, bin, bins[bin], nb);

    std::cout << "  Bin " << BinNames[b] << ": wrote " << bins[bin].size() << " queries" << std::endl;
  }

  std::cout << "\n================================================================================" << std::endl;
  std::cout << "Done!" << std::endl;
  std::cout << "================================================================================" << std::endl;

  delete[] base_vecs;
  delete[] query_vecs;
  return 0;
}
