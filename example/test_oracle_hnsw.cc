#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>
#include <map>

#include "../hnswlib/hnswlib/hnswalg.h"
#include "../hnswlib/hnswlib/space_l2.h"
#include "../hnswlib/hnswlib/space_ip.h"

// Define types from lssg
namespace lssg {
typedef uint32_t label_t;
typedef uint16_t labelsetmember_t;
using labelset_t = std::vector<labelsetmember_t>;
}  // namespace lssg

// Utility functions copied from bench_utils.hh
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
          throw;
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
  std::cout << "Loaded ground truth: " << all_gt.size() << " queries" << std::endl;
  return all_gt;
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

}  // namespace benchmark

// Helper function to extract base filename without path and extension
std::string extractBaseFilename(const std::string &filepath)
{
  // Find last slash
  size_t      last_slash = filepath.find_last_of("/\\");
  std::string filename   = (last_slash == std::string::npos) ? filepath : filepath.substr(last_slash + 1);

  // Remove extension
  size_t last_dot = filename.find_last_of('.');
  if (last_dot != std::string::npos) {
    filename = filename.substr(0, last_dot);
  }

  return filename;
}

// Helper function to create directory if it doesn't exist
bool ensureDirectoryExists(const std::string &dir)
{
  struct stat st;
  if (stat(dir.c_str(), &st) == 0) {
    return S_ISDIR(st.st_mode);
  }

  // Try to create directory
  std::filesystem::create_directories(dir);
  return true;
}

// Hash function for label sets
struct LabelSetHash
{
  std::size_t operator()(const std::vector<uint16_t> &v) const
  {
    std::size_t seed = v.size();
    for (auto &i : v) {
      seed ^= std::hash<uint16_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

// Filter scenario types
enum class FilterScenario
{
  EQUALITY,
  CONTAINMENT,
  OVERLAP
};

// Convert scenario to string
std::string scenarioToString(FilterScenario scenario)
{
  switch (scenario) {
    case FilterScenario::EQUALITY: return "equality";
    case FilterScenario::CONTAINMENT: return "containment";
    case FilterScenario::OVERLAP: return "overlap";
    default: return "unknown";
  }
}

// Check if base_labels satisfies query_labels under given scenario
bool satisfiesFilter(
    const lssg::labelset_t &query_labels, const lssg::labelset_t &base_labels, FilterScenario scenario)
{
  switch (scenario) {
    case FilterScenario::EQUALITY: return base_labels == query_labels;

    case FilterScenario::CONTAINMENT:
      // base_labels must be superset of query_labels (contain all query labels)
      return std::includes(base_labels.begin(), base_labels.end(), query_labels.begin(), query_labels.end());

    case FilterScenario::OVERLAP:
      // base_labels must have at least one common element with query_labels
      return std::any_of(query_labels.begin(), query_labels.end(), [&base_labels](lssg::labelsetmember_t label) {
        return std::binary_search(base_labels.begin(), base_labels.end(), label);
      });

    default: return false;
  }
}

// Oracle HNSW manager
template <typename dist_t>
class OracleHNSWManager
{
public:
  struct OracleKey
  {
    lssg::labelset_t query_labels;
    FilterScenario     scenario;

    bool operator==(const OracleKey &other) const
    {
      return query_labels == other.query_labels && scenario == other.scenario;
    }
  };

  struct OracleKeyHash
  {
    std::size_t operator()(const OracleKey &key) const
    {
      LabelSetHash          hash_fn;
      std::vector<uint16_t> temp_vec(key.query_labels.begin(), key.query_labels.end());
      std::size_t           h1 = hash_fn(temp_vec);
      std::size_t           h2 = std::hash<int>{}(static_cast<int>(key.scenario));
      return h1 ^ (h2 << 1);
    }
  };

  struct OracleIndex
  {
    std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>> index;
    std::vector<size_t>                               filtered_ids;  // Original IDs of vectors in this index
    size_t                                            build_time_ms;

    OracleIndex() : build_time_ms(0) {}
  };

  OracleHNSWManager(hnswlib::SpaceInterface<dist_t> *space, size_t M, size_t ef_construction,
      const std::string &index_dir = "", const std::string &base_vec_name = "", size_t max_memory_indices = 10,
      bool verbose = false)
      : space_(space),
        M_(M),
        ef_construction_(ef_construction),
        index_dir_(index_dir),
        base_vec_name_(base_vec_name),
        max_memory_indices_(max_memory_indices),
        verbose_(verbose)
  {
    if (!index_dir_.empty()) {
      ensureDirectoryExists(index_dir_);
    }
  }

  // Get or build oracle index for given query and scenario
  OracleIndex *getOrBuildIndex(const OracleKey &key, const std::vector<lssg::labelset_t> &base_labels,
      const float *base_data, size_t data_dim, size_t num_base)
  {
    auto it = oracle_indices_.find(key);
    if (it != oracle_indices_.end()) {
      cache_hits_++;
      return &it->second;
    }

    cache_misses_++;

    // Try to load from disk first
    OracleIndex loaded_idx;
    if (loadIndexFromDisk(key, loaded_idx, num_base)) {
      evictIfNeeded();
      auto &stored_idx = oracle_indices_[key];
      stored_idx       = std::move(loaded_idx);
      if (verbose_) {
        std::cout << "Loaded oracle HNSW from disk for query labels {";
        for (size_t i = 0; i < key.query_labels.size(); i++) {
          std::cout << key.query_labels[i];
          if (i < key.query_labels.size() - 1)
            std::cout << ",";
        }
        std::cout << "} [" << scenarioToString(key.scenario) << "] with " << stored_idx.filtered_ids.size()
                  << " vectors" << std::endl;
      }
      return &stored_idx;
    }

    // Find all vectors that satisfy the filter
    std::vector<size_t> filtered_ids;
    for (size_t i = 0; i < num_base; i++) {
      if (satisfiesFilter(key.query_labels, base_labels[i], key.scenario)) {
        filtered_ids.push_back(i);
      }
    }

    if (filtered_ids.empty()) {
      std::cout << "Warning: No vectors satisfy filter for query labels {";
      for (size_t i = 0; i < key.query_labels.size(); i++) {
        std::cout << key.query_labels[i];
        if (i < key.query_labels.size() - 1)
          std::cout << ",";
      }
      std::cout << "} with scenario " << scenarioToString(key.scenario) << std::endl;
      return nullptr;
    }

    // Build HNSW index over filtered vectors
    auto start = std::chrono::high_resolution_clock::now();

    auto hnsw = std::make_unique<hnswlib::HierarchicalNSW<dist_t>>(space_, filtered_ids.size(), M_, ef_construction_);

// Add filtered vectors to index
#pragma omp parallel for
    for (size_t idx = 0; idx < filtered_ids.size(); idx++) {
      size_t       original_id = filtered_ids[idx];
      const float *vec_data    = base_data + original_id * data_dim;
      hnsw->addPoint(vec_data, original_id);
    }

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Store the index
    OracleIndex oracle_idx;
    oracle_idx.index         = std::move(hnsw);
    oracle_idx.filtered_ids  = std::move(filtered_ids);
    oracle_idx.build_time_ms = duration.count();

    // Save to disk if directory is configured
    if (!index_dir_.empty()) {
      saveIndexToDisk(key, oracle_idx);
    }

    evictIfNeeded();

    auto &stored_idx = oracle_indices_[key];
    stored_idx       = std::move(oracle_idx);

    if (verbose_) {
      std::cout << "Built oracle HNSW for query labels {";
      for (size_t i = 0; i < key.query_labels.size(); i++) {
        std::cout << key.query_labels[i];
        if (i < key.query_labels.size() - 1)
          std::cout << ",";
      }
      std::cout << "} [" << scenarioToString(key.scenario) << "] with " << stored_idx.filtered_ids.size()
                << " vectors in " << stored_idx.build_time_ms << "ms" << std::endl;
    }

    return &stored_idx;
  }

  // Clear all in-memory indices (for single-index mode)
  void clearAllIndices() { oracle_indices_.clear(); }

  // Delete index files from disk for a specific key
  void deleteIndexFromDisk(const OracleKey &key)
  {
    if (index_dir_.empty()) {
      return;
    }

    try {
      std::string filename = generateIndexFilename(key);
      std::string ids_filename = filename + ".ids";
      
      // Remove both the index file and the ids file
      std::remove(filename.c_str());
      std::remove(ids_filename.c_str());
      
      if (verbose_) {
        std::cout << "Deleted index from disk: " << filename << std::endl;
      }
    } catch (const std::exception &e) {
      if (verbose_) {
        std::cerr << "Failed to delete index: " << e.what() << std::endl;
      }
    }
  }

  // Get statistics
  size_t getCacheHits() const { return cache_hits_; }
  size_t getCacheMisses() const { return cache_misses_; }

  void printStatistics() const
  {
    std::cout << "\n=== Oracle HNSW Statistics ===" << std::endl;
    std::cout << "Total oracle indices: " << oracle_indices_.size() << std::endl;
    std::cout << "Cache hits: " << cache_hits_ << std::endl;
    std::cout << "Cache misses: " << cache_misses_ << std::endl;

    size_t total_build_time       = 0;
    size_t total_filtered_vectors = 0;
    for (const auto &[key, idx] : oracle_indices_) {
      total_build_time += idx.build_time_ms;
      total_filtered_vectors += idx.filtered_ids.size();
    }

    std::cout << "Total build time: " << total_build_time << "ms" << std::endl;
    std::cout << "Average filtered vectors per index: "
              << (oracle_indices_.empty() ? 0 : total_filtered_vectors / oracle_indices_.size()) << std::endl;
    std::cout << "===============================" << std::endl;
  }

private:
  // Generate filename for storing/loading index
  std::string generateIndexFilename(const OracleKey &key) const
  {
    std::stringstream ss;
    for (size_t i = 0; i < key.query_labels.size(); i++) {
      ss << key.query_labels[i];
      if (i < key.query_labels.size() - 1)
        ss << "_";
    }
    std::string labelset_str = ss.str();

    return index_dir_ + "/" + base_vec_name_ + "_" + scenarioToString(key.scenario) + "_" + labelset_str + ".hnsw";
  }

  // Save index to disk
  bool saveIndexToDisk(const OracleKey &key, const OracleIndex &idx)
  {
    if (index_dir_.empty() || !idx.index) {
      return false;
    }

    try {
      std::string filename = generateIndexFilename(key);
      idx.index->saveIndex(filename);

      // Save filtered_ids separately
      std::string   ids_filename = filename + ".ids";
      std::ofstream ids_file(ids_filename, std::ios::binary);
      size_t        ids_size = idx.filtered_ids.size();
      ids_file.write(reinterpret_cast<const char *>(&ids_size), sizeof(size_t));
      ids_file.write(reinterpret_cast<const char *>(idx.filtered_ids.data()), ids_size * sizeof(size_t));
      ids_file.close();

      if (verbose_) {
        std::cout << "Saved index to disk: " << filename << std::endl;
      }
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Failed to save index: " << e.what() << std::endl;
      return false;
    }
  }

  // Load index from disk
  bool loadIndexFromDisk(const OracleKey &key, OracleIndex &idx, size_t max_elements)
  {
    if (index_dir_.empty()) {
      return false;
    }

    try {
      std::string filename = generateIndexFilename(key);

      // Check if file exists
      std::ifstream test_file(filename);
      if (!test_file.good()) {
        return false;
      }
      test_file.close();

      auto hnsw = std::make_unique<hnswlib::HierarchicalNSW<dist_t>>(space_, filename, false, max_elements);

      // Load filtered_ids
      std::string   ids_filename = filename + ".ids";
      std::ifstream ids_file(ids_filename, std::ios::binary);
      if (!ids_file.good()) {
        return false;
      }

      size_t ids_size;
      ids_file.read(reinterpret_cast<char *>(&ids_size), sizeof(size_t));
      idx.filtered_ids.resize(ids_size);
      ids_file.read(reinterpret_cast<char *>(idx.filtered_ids.data()), ids_size * sizeof(size_t));
      ids_file.close();

      idx.index         = std::move(hnsw);
      idx.build_time_ms = 0;  // Loaded from disk, not built

      if (verbose_) {
        std::cout << "Loaded index from disk: " << filename << std::endl;
      }
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Failed to load index: " << e.what() << std::endl;
      return false;
    }
  }

  // Evict least recently used index if memory limit reached
  void evictIfNeeded()
  {
    if (oracle_indices_.size() <= max_memory_indices_) {
      return;
    }

    // Find LRU index (we'll evict the first one for simplicity)
    // In a more sophisticated implementation, you'd track access times
    auto it = oracle_indices_.begin();
    if (it != oracle_indices_.end()) {
      saveIndexToDisk(it->first, it->second);
      oracle_indices_.erase(it);
      if (verbose_) {
        std::cout << "Evicted one index from memory (total in memory: " << oracle_indices_.size() << ")" << std::endl;
      }
    }
  }

  hnswlib::SpaceInterface<dist_t>                          *space_;
  size_t                                                    M_;
  size_t                                                    ef_construction_;
  std::unordered_map<OracleKey, OracleIndex, OracleKeyHash> oracle_indices_;
  size_t                                                    cache_hits_   = 0;
  size_t                                                    cache_misses_ = 0;
  std::string                                               index_dir_;
  std::string                                               base_vec_name_;
  size_t                                                    max_memory_indices_;
  bool                                                      verbose_;
};

// Helper structure for grouping queries by label set
struct QueryGroup
{
  lssg::labelset_t  labelset;
  std::vector<size_t> query_ids;
};

// Group queries by their label sets for efficient processing
std::vector<QueryGroup> groupQueriesByLabelSet(const std::vector<lssg::labelset_t> &query_labels)
{
  std::unordered_map<lssg::labelset_t, std::vector<size_t>, LabelSetHash> labelset_to_queries;

  // Group queries by label set
  for (size_t i = 0; i < query_labels.size(); i++) {
    labelset_to_queries[query_labels[i]].push_back(i);
  }

  // Convert to vector for ordered processing
  std::vector<QueryGroup> groups;
  groups.reserve(labelset_to_queries.size());

  for (auto &[labelset, query_ids] : labelset_to_queries) {
    QueryGroup group;
    group.labelset  = labelset;
    group.query_ids = std::move(query_ids);
    groups.push_back(std::move(group));
  }

  // Sort groups by size (largest first) to process common label sets first
  std::sort(groups.begin(), groups.end(), [](const QueryGroup &a, const QueryGroup &b) {
    return a.query_ids.size() > b.query_ids.size();
  });

  std::cout << "Grouped " << query_labels.size() << " queries into " << groups.size() << " label set groups"
            << std::endl;

  return groups;
}

// Main test function
template <typename dist_t>
void runOracleHNSWTest(const std::string &base_file, const std::string &query_file, const std::string &base_label_file,
    const std::string &query_label_file, FilterScenario scenario, const std::string &groundtruth_file,
    hnswlib::SpaceInterface<dist_t> *space, size_t M, size_t ef_construction, size_t topk,
    const std::string &index_dir = "", size_t max_memory_indices = 10)
{

  std::cout << "\n========================================" << std::endl;
  std::cout << "Oracle HNSW Test" << std::endl;
  std::cout << "Scenario: " << scenarioToString(scenario) << std::endl;
  std::cout << "M: " << M << ", ef_construction: " << ef_construction << std::endl;
  if (!index_dir.empty()) {
    std::cout << "Index directory: " << index_dir << std::endl;
    std::cout << "Max memory indices: " << max_memory_indices << std::endl;
  }
  std::cout << "========================================\n" << std::endl;

  // Load base vectors
  size_t num_base, data_dim;
  float *base_data = benchmark::fvecs_read(base_file, data_dim, num_base);

  // Load query vectors
  size_t num_query, query_dim;
  float *query_data = benchmark::fvecs_read(query_file, query_dim, num_query);

  if (data_dim != query_dim) {
    std::cerr << "Error: Base and query dimensions do not match!" << std::endl;
    delete[] base_data;
    delete[] query_data;
    return;
  }

  // Load base labels
  auto base_labels = benchmark::LoadLabelSets(base_label_file);
  if (base_labels.size() != num_base) {
    std::cerr << "Error: Base labels count (" << base_labels.size() << ") does not match base vectors count ("
              << num_base << ")" << std::endl;
    delete[] base_data;
    delete[] query_data;
    return;
  }

  // Load query labels
  auto query_labels = benchmark::LoadLabelSets(query_label_file);
  if (query_labels.size() != num_query) {
    std::cerr << "Warning: Query labels count (" << query_labels.size() << ") does not match query vectors count ("
              << num_query << ")" << std::endl;
    num_query = std::min(query_labels.size(), num_query);
    query_labels.resize(num_query);
  }

  // Load ground truth
  auto gt_data = benchmark::LoadGroundTruth(groundtruth_file);
  std::cout << "Loaded ground truth" << std::endl;

  // Extract base filename for index naming
  std::string base_vec_name = extractBaseFilename(base_file);

  // Create oracle manager
  OracleHNSWManager<dist_t> oracle_manager(space, M, ef_construction, index_dir, base_vec_name, max_memory_indices);

  // Define ef_search values to test (similar to search_po_label.cc)
  std::vector<size_t> efs_list = {1700,
      1400,
      1100,
      1000,
      900,
      800,
      700,
      600,
      500,
      400,
      300,
      250,
      200,
      180,
      160,
      140,
      120,
      100,
      90,
      80,
      70,
      60,
      55,
      50,
      45,
      40,
      35,
      30,
      25,
      20,
      15,
      10};

  // Structure to accumulate statistics for each ef_search
  struct EfStatistics
  {
    double total_recall                = 0.0;
    double total_query_time_us         = 0.0;
    double total_distance_computations = 0.0;
    double total_index_size            = 0.0;
    size_t valid_queries               = 0;
  };

  std::map<size_t, EfStatistics> ef_stats_map;

  // Initialize statistics for all ef values
  for (auto ef : efs_list) {
    ef_stats_map[ef] = EfStatistics();
  }

  // Group queries by label set for efficient processing
  auto query_groups = groupQueriesByLabelSet(query_labels);

  std::cout << "\nProcessing query groups (group-by-group to minimize memory usage)..." << std::endl;

  // Outer loop: iterate through query groups
  for (size_t g = 0; g < query_groups.size(); g++) {
    const auto &group = query_groups[g];

    // Single line progress update with \r
    std::cout << "\rProcessing group " << (g + 1) << "/" << query_groups.size() << " (" << group.query_ids.size()
              << " queries)..." << std::flush;

    // Create oracle key for this group
    typename OracleHNSWManager<dist_t>::OracleKey key;
    key.query_labels = group.labelset;
    key.scenario     = scenario;

    // Build or load oracle index for this group
    auto oracle_idx = oracle_manager.getOrBuildIndex(key, base_labels, base_data, data_dim, num_base);

    if (!oracle_idx || !oracle_idx->index) {
      std::cout << "\rWarning: Failed to get index for group " << (g + 1) << ", skipping...            " << std::endl;
      continue;
    }

    // Process all queries in this group
    for (size_t q_idx : group.query_ids) {
      // Skip if no ground truth for this query
      if (q_idx >= gt_data.size() || gt_data[q_idx].empty()) {
        continue;
      }

      const float *query_vec = query_data + q_idx * data_dim;

      // Inner loop: test all ef_search values for this query
      for (auto ef_search : efs_list) {
        // Set ef for search
        oracle_idx->index->setEf(ef_search);

        // Reset distance computation counter
        size_t dist_comps_before = oracle_idx->index->metric_distance_computations;

        auto query_start = std::chrono::high_resolution_clock::now();
        auto result      = oracle_idx->index->searchKnn(query_vec, topk);
        auto query_end   = std::chrono::high_resolution_clock::now();

        size_t dist_comps_after = oracle_idx->index->metric_distance_computations;
        size_t dist_comps       = dist_comps_after - dist_comps_before;

        auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);

        // Calculate recall using benchmark utility
        std::vector<lssg::label_t> result_ids;
        while (!result.empty()) {
          auto [dist, label] = result.top();
          result.pop();
          result_ids.push_back(label);
        }

        // Reverse to get nearest first (priority queue returns farthest first)
        std::reverse(result_ids.begin(), result_ids.end());

        double recall = benchmark::CalculateRecall(gt_data[q_idx], result_ids);

        // Accumulate statistics for this ef_search
        ef_stats_map[ef_search].total_recall += recall;
        ef_stats_map[ef_search].total_query_time_us += query_duration.count();
        ef_stats_map[ef_search].total_distance_computations += dist_comps;
        ef_stats_map[ef_search].total_index_size += oracle_idx->filtered_ids.size();
        ef_stats_map[ef_search].valid_queries++;
      }
    }

    // Clear this index from memory after processing the group
    // This avoids keeping unnecessary indices in memory
    oracle_manager.clearAllIndices();
    
    // Delete the index from disk to save storage space
    oracle_manager.deleteIndexFromDisk(key);
  }

  std::cout << "\rAll query groups processed (" << query_groups.size() << " groups)                    " << std::endl;

  // CSV header
  std::cout << "ef,recall,qps,avg_distance_computations,avg_index_size" << std::endl;

  // Output results for each ef_search value
  for (auto ef_search : efs_list) {
    const auto &stats = ef_stats_map[ef_search];

    if (stats.valid_queries == 0) {
      continue;
    }

    // Calculate averages
    double avg_recall                = stats.total_recall / stats.valid_queries;
    double avg_query_time_us         = stats.total_query_time_us / stats.valid_queries;
    double avg_distance_computations = stats.total_distance_computations / stats.valid_queries;
    double avg_index_size            = stats.total_index_size / stats.valid_queries;
    double qps                       = avg_query_time_us > 0 ? (1000000.0 / avg_query_time_us) : 0.0;

    // Output in CSV format
    std::cout << ef_search << "," << avg_recall << "," << qps << "," << avg_distance_computations << ","
              << avg_index_size << std::endl;
  }

  oracle_manager.printStatistics();

  // Clean up
  delete[] base_data;
  delete[] query_data;
}

int main(int argc, char **argv)
{
  if (argc < 8) {
    std::cout << "Usage: " << argv[0] << " <base_file> <query_file> <base_label_file> <query_label_file>"
              << " <scenario:equality|containment|overlap> <groundtruth_file> <metric:l2|ip>"
              << " [M=16] [ef_construction=128] [topk=10] [index_dir=''] [max_memory_indices=10]" << std::endl;
    return 1;
  }

  std::string base_file        = argv[1];
  std::string query_file       = argv[2];
  std::string base_label_file  = argv[3];
  std::string query_label_file = argv[4];
  std::string scenario_str     = argv[5];
  std::string groundtruth_file = argv[6];
  std::string metric           = argv[7];

  size_t      M                  = argc > 8 ? std::atoi(argv[8]) : 16;
  size_t      ef_construction    = argc > 9 ? std::atoi(argv[9]) : 128;
  size_t      topk               = argc > 10 ? std::atoi(argv[10]) : 10;
  std::string index_dir          = argc > 11 ? argv[11] : "";
  size_t      max_memory_indices = argc > 12 ? std::atoi(argv[12]) : 10;

  // Parse scenario
  FilterScenario scenario;
  if (scenario_str == "equality") {
    scenario = FilterScenario::EQUALITY;
  } else if (scenario_str == "containment") {
    scenario = FilterScenario::CONTAINMENT;
  } else if (scenario_str == "overlap") {
    scenario = FilterScenario::OVERLAP;
  } else {
    std::cerr << "Error: Unknown scenario '" << scenario_str << "'" << std::endl;
    return 1;
  }

  // Get data dimension from base file
  size_t num_base, data_dim;
  float *temp_data = benchmark::fvecs_read(base_file, data_dim, num_base);
  delete[] temp_data;

  // Create space
  if (metric == "l2") {
    hnswlib::L2Space space(data_dim);
    runOracleHNSWTest<float>(base_file,
        query_file,
        base_label_file,
        query_label_file,
        scenario,
        groundtruth_file,
        &space,
        M,
        ef_construction,
        topk,
        index_dir,
        max_memory_indices);
  } else if (metric == "ip") {
    hnswlib::InnerProductSpace space(data_dim);
    runOracleHNSWTest<float>(base_file,
        query_file,
        base_label_file,
        query_label_file,
        scenario,
        groundtruth_file,
        &space,
        M,
        ef_construction,
        topk,
        index_dir,
        max_memory_indices);
  } else {
    std::cerr << "Error: Unknown metric '" << metric << "'" << std::endl;
    return 1;
  }

  return 0;
}
