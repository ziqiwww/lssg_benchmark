#include "../lssg/poindex.hh"
#include "bench_utils.hh"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <set>
#include <unordered_set>
#include <map>

struct QueryScenario
{
  lssg::LabelFilterConfig::Type                type;
  std::string                                    name;
  std::string                                    gt_file;
  std::vector<std::vector<lssg::label_t>>      gt_data;
};

struct LowRecallQueryStats
{
  size_t query_id;
  size_t ef;
  float recall;
  size_t matching_vectors;
  size_t unique_labelsets;
};

int main(int argc, char **argv)
{
  std::string query_vec_file, query_label_file, base_vec_file, base_label_file;
  std::string index_location, scope_location, space;
  std::string gt_containment, gt_equality, gt_overlap;
  size_t      k = 10;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--query_vec") == 0) {
      query_vec_file = argv[++i];
    } else if (strcmp(argv[i], "--query_labelset") == 0) {
      query_label_file = argv[++i];
    } else if (strcmp(argv[i], "--basevec") == 0) {
      base_vec_file = argv[++i];
    } else if (strcmp(argv[i], "--base_labelset") == 0) {
      base_label_file = argv[++i];
    } else if (strcmp(argv[i], "--k") == 0) {
      k = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--index_location") == 0) {
      index_location = argv[++i];
    } else if (strcmp(argv[i], "--scope_location") == 0) {
      scope_location = argv[++i];
    } else if (strcmp(argv[i], "--space") == 0) {
      space = argv[++i];
    } else if (strcmp(argv[i], "--gt_containment") == 0) {
      gt_containment = argv[++i];
    } else if (strcmp(argv[i], "--gt_equality") == 0) {
      gt_equality = argv[++i];
    } else if (strcmp(argv[i], "--gt_overlap") == 0) {
      gt_overlap = argv[++i];
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argv[i]));
    }
  }

  if (query_vec_file.empty() || query_label_file.empty() || index_location.empty() || scope_location.empty() ||
      space.empty()) {
    throw std::runtime_error("Missing required arguments for search_po_label");
  }

  size_t d = 0, nq = 0;
  auto   query_vecs = benchmark::fvecs_read(query_vec_file, d, nq);
  auto query_labels = benchmark::LoadLabelSets(query_label_file);
  if (query_labels.size() != nq) {
    nq = std::min(query_labels.size(), nq);
    query_labels.resize(nq);
  }

  std::vector<lssg::labelset_t> base_labelsets;
  float                          *base_vecs = nullptr;
  size_t                          nb = 0, base_dim = 0;
  if (!base_label_file.empty()) {
    base_labelsets = benchmark::LoadLabelSets(base_label_file);
    if (base_vec_file.empty()) {
      delete[] query_vecs;
      throw std::runtime_error("basevec path is required when base_labelset is provided");
    }
    base_vecs = benchmark::fvecs_read(base_vec_file, base_dim, nb);
    if (base_dim != d) {
      delete[] query_vecs;
      delete[] base_vecs;
      throw std::runtime_error("Base vector dimension does not match query dimension");
    }
    if (nb != base_labelsets.size()) {
      delete[] query_vecs;
      delete[] base_vecs;
      throw std::runtime_error("Base vectors count does not match label set count");
    }
  }

  lssg::PoIndex<lssg::labelset_t, float, lssg::LabelSetScopeKernel> index(index_location, scope_location, space);

  // Create scenarios based on which GT files were provided
  std::vector<QueryScenario> scenarios;
  
  if (!gt_containment.empty()) {
    scenarios.push_back({lssg::LabelFilterConfig::LABEL_CONTAINMENT, "containment", gt_containment, {}});
  }
  
  if (!gt_equality.empty()) {
    scenarios.push_back({lssg::LabelFilterConfig::LABEL_EQUALITY, "equality", gt_equality, {}});
  }
  
  if (!gt_overlap.empty()) {
    scenarios.push_back({lssg::LabelFilterConfig::LABEL_OVERLAP, "overlap", gt_overlap, {}});
  }

  // Load ground truth data for all scenarios upfront
  for (auto &scenario : scenarios) {
    if (!scenario.gt_file.empty()) {
      scenario.gt_data = benchmark::LoadGroundTruth(scenario.gt_file);
    } else if (!base_labelsets.empty() && base_vecs != nullptr) {
      scenario.gt_data = benchmark::GenLabelSetGT(nb, nq, d, k, base_vecs, query_vecs, base_labelsets, query_labels,
          scenario.type, space);
    } else {
      std::cout << "Skip " << scenario.name
                << " evaluation due to missing ground truth and base data." << std::endl;
      scenario.gt_data.clear();
    }
  }

  // 10000, 5000, 3000, 2500, 2000,
  std::vector<size_t> efs_list = {20000, 15000, 10000, 5000, 2500, 2000, 1700,1400,1100,1000,900,800,700,600,500,400,300,250,200,180,
    160,140,120,100,90,80,70,60,55,50,45,40,35,30,25,20,15,10};
  
  // std::vector<size_t> efs_list = {1700};

  // std::vector<size_t> efs_list = {1700,1400,};

  // CSV header - only print once at the beginning
  std::cout << "ef,recall,qps,dist_comp,hops" << std::endl;

  // REFACTORED: Test all EFS values for each scenario
  for (const auto &scenario : scenarios) {
    if (scenario.gt_data.empty()) {
      continue;
    }
    
    lssg::LabelFilterConfig cfg;
    cfg.type_ = scenario.type;
    
    std::vector<LowRecallQueryStats> all_low_recall_stats;
    
    for (auto ef : efs_list) {
      std::vector<std::vector<lssg::label_t>> results(nq);
      std::vector<std::vector<std::pair<float, lssg::label_t>>> results_with_dist(nq);
      double total_time = 0.0;
      index.metric_dist_comps_ = 0;
      index.metric_hops_ = 0;

      for (size_t i = 0; i < nq; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto res = index.searchKNN(query_vecs + i * d, ef, k, query_labels[i], cfg);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double>(end - start).count();
        for (const auto &pair : res) {
          results[i].emplace_back(pair.second);
          results_with_dist[i].emplace_back(pair.first, pair.second);
        }
      }
      
      // // Calculate per-query recall and collect low recall statistics
      // if (!base_labelsets.empty() && nb > 0) {
      //   for (size_t i = 0; i < nq; ++i) {
      //     float query_recall = benchmark::CalculateRecall(scenario.gt_data[i], results[i]);
          
      //     if (query_recall < 0.5f) {
      //       // Count matching vectors and unique labelsets
      //       size_t matching_vectors = 0;
      //       std::set<std::string> unique_labelsets_str;
            
      //       for (size_t j = 0; j < nb; ++j) {
      //         if (benchmark::LabelFilterMatch(base_labelsets[j], query_labels[i], cfg.type_)) {
      //           matching_vectors++;
                
      //           // Convert labelset to string for uniqueness check
      //           std::string labelset_str;
      //           for (auto label : base_labelsets[j]) {
      //             labelset_str += std::to_string(label) + ",";
      //           }
      //           unique_labelsets_str.insert(labelset_str);
      //         }
      //       }
            
      //       LowRecallQueryStats stats;
      //       stats.query_id = i;
      //       stats.ef = ef;
      //       stats.recall = query_recall;
      //       stats.matching_vectors = matching_vectors;
      //       stats.unique_labelsets = unique_labelsets_str.size();
      //       all_low_recall_stats.push_back(stats);
      //     }
      //   }
      // }

      // // Calculate per-query recall and log failures (only if base data is available)
      // if (!base_labelsets.empty() && nb > 0) {
      //   std::vector<float> per_query_recall(nq);
      //   for (size_t i = 0; i < nq; ++i) {
      //     per_query_recall[i] = benchmark::CalculateRecall(scenario.gt_data[i], results[i]);
          
      //     // Log queries with recall < 0.9
      //     if (per_query_recall[i] < 0.9f) {
      //       // Count data points and unique labelsets that satisfy the filter
      //       size_t matching_data_count = 0;
      //       std::unordered_set<lssg::labelset_t> unique_matching_labelsets;
            
      //       for (size_t j = 0; j < nb; ++j) {
      //         bool satisfies_filter = benchmark::LabelFilterMatch(base_labelsets[j], query_labels[i], cfg.type_);
              
      //         if (satisfies_filter) {
      //           matching_data_count++;
      //           unique_matching_labelsets.insert(base_labelsets[j]);
      //         }
      //       }
            
      //       // Print header
      //       std::cout << "\n========== LOW RECALL QUERY ==========\n";
      //       std::cout << "Scenario: " << scenario.name << " | EF: " << ef << " | Query ID: " << i 
      //                 << " | Recall: " << per_query_recall[i] << "\n";
            
      //       // Print query labelset
      //       std::cout << "Query Labels: {";
      //       for (size_t j = 0; j < query_labels[i].size(); ++j) {
      //         if (j > 0) std::cout << ", ";
      //         std::cout << query_labels[i][j];
      //       }
      //       std::cout << "} (size=" << query_labels[i].size() << ")\n";
      //       std::cout << "Matching Data: " << matching_data_count 
      //                 << " | Unique Labelsets: " << unique_matching_labelsets.size() << "\n";
            
      //       // Print KNN results (up to 20)
      //       std::cout << "\nKNN Results (" << results[i].size() << " returned):\n";
      //       for (size_t j = 0; j < std::min(results_with_dist[i].size(), size_t(20)); ++j) {
      //         float dist = results_with_dist[i][j].first;
      //         lssg::label_t vid = results_with_dist[i][j].second;
      //         std::cout << "  [" << j << "] id=" << vid << " dist=" << dist << " labels={";
      //         for (size_t l = 0; l < base_labelsets[vid].size() && l < 10; ++l) {
      //           if (l > 0) std::cout << ",";
      //           std::cout << base_labelsets[vid][l];
      //         }
      //         if (base_labelsets[vid].size() > 10) std::cout << "...";
      //         std::cout << "}\n";
      //       }
            
      //       // Print ground truth (up to 20)
      //       std::cout << "\nGround Truth (" << scenario.gt_data[i].size() << " expected):\n";
      //       for (size_t j = 0; j < std::min(scenario.gt_data[i].size(), size_t(20)); ++j) {
      //         lssg::label_t vid = scenario.gt_data[i][j];
      //         bool found_in_results = std::find(results[i].begin(), results[i].end(), vid) != results[i].end();
              
      //         // Compute distance from query to this ground truth point
      //         float gt_dist = 0.0f;
      //         if (base_vecs != nullptr) {
      //           const float* query_vec = query_vecs + i * d;
      //           const float* base_vec = base_vecs + vid * d;
                
      //           if (space == "l2") {
      //             for (size_t dim = 0; dim < d; ++dim) {
      //               float diff = query_vec[dim] - base_vec[dim];
      //               gt_dist += diff * diff;
      //             }
      //           } else if (space == "ip") {
      //             for (size_t dim = 0; dim < d; ++dim) {
      //               gt_dist += query_vec[dim] * base_vec[dim];
      //             }
      //             gt_dist = 1.0f - gt_dist;  // Convert to distance
      //           }
      //         }
              
      //         std::cout << "  [" << j << "] id=" << vid << " dist=" << gt_dist
      //                   << (found_in_results ? " [FOUND]" : " [MISS]") << " labels={";
      //         for (size_t l = 0; l < base_labelsets[vid].size() && l < 10; ++l) {
      //           if (l > 0) std::cout << ",";
      //           std::cout << base_labelsets[vid][l];
      //         }
      //         if (base_labelsets[vid].size() > 10) std::cout << "...";
      //         std::cout << "}\n";
      //       }
            
      //       std::cout << "======================================\n" << std::endl;
      //     }
      //   }
      // }

      float recall = benchmark::CalculateRecall(scenario.gt_data, results);
      double qps = nq / total_time;
      
      // Output in CSV format for easier plotting
      std::cout << ef << "," << recall << "," << qps << "," 
                << static_cast<double>(index.metric_dist_comps_) / nq << "," 
                << static_cast<double>(index.metric_hops_) / nq << std::endl;
    }
    
    // Print comprehensive low recall statistics summary
    // if (!all_low_recall_stats.empty() && !base_labelsets.empty()) {
    //   std::cout << "\n========== LOW RECALL (<0.5) STATISTICS FOR " << scenario.name << " ==========\n";
    //   std::cout << "Total low recall query instances: " << all_low_recall_stats.size() << "\n";
      
    //   // Group by EF and calculate percentages
    //   std::map<size_t, size_t> ef_low_recall_counts;
    //   std::map<size_t, size_t> ef_total_queries;
      
    //   for (const auto& stat : all_low_recall_stats) {
    //     ef_low_recall_counts[stat.ef]++;
    //   }
      
    //   for (auto ef : efs_list) {
    //     ef_total_queries[ef] = nq;
    //   }
      
    //   std::cout << "\nLow Recall Query Percentage by EF:\n";
    //   for (auto ef : efs_list) {
    //     if (ef_low_recall_counts.count(ef)) {
    //       float pct = 100.0f * ef_low_recall_counts[ef] / nq;
    //       std::cout << "  EF=" << ef << ": " << ef_low_recall_counts[ef] << "/" << nq 
    //                 << " (" << pct << "%)\n";
    //     }
    //   }
      
    //   // Calculate percentiles for matching vectors and unique labelsets
    //   std::vector<size_t> matching_vectors_values;
    //   std::vector<size_t> unique_labelsets_values;
    //   std::vector<float> recall_values;
      
    //   for (const auto& stat : all_low_recall_stats) {
    //     matching_vectors_values.push_back(stat.matching_vectors);
    //     unique_labelsets_values.push_back(stat.unique_labelsets);
    //     recall_values.push_back(stat.recall);
    //   }
      
    //   std::sort(matching_vectors_values.begin(), matching_vectors_values.end());
    //   std::sort(unique_labelsets_values.begin(), unique_labelsets_values.end());
    //   std::sort(recall_values.begin(), recall_values.end());
      
    //   auto get_percentile = [](const std::vector<size_t>& sorted_vec, float percentile) -> size_t {
    //     size_t idx = static_cast<size_t>(percentile / 100.0f * (sorted_vec.size() - 1));
    //     return sorted_vec[idx];
    //   };
      
    //   auto get_percentile_float = [](const std::vector<float>& sorted_vec, float percentile) -> float {
    //     size_t idx = static_cast<size_t>(percentile / 100.0f * (sorted_vec.size() - 1));
    //     return sorted_vec[idx];
    //   };
      
    //   std::cout << "\nMatching Vectors Distribution (percentiles):\n";
    //   std::cout << "  Min (p0):  " << matching_vectors_values.front() << "\n";
    //   std::cout << "  p25:       " << get_percentile(matching_vectors_values, 25) << "\n";
    //   std::cout << "  Median:    " << get_percentile(matching_vectors_values, 50) << "\n";
    //   std::cout << "  p75:       " << get_percentile(matching_vectors_values, 75) << "\n";
    //   std::cout << "  Max (p100):" << matching_vectors_values.back() << "\n";
      
    //   std::cout << "\nUnique Labelsets Distribution (percentiles):\n";
    //   std::cout << "  Min (p0):  " << unique_labelsets_values.front() << "\n";
    //   std::cout << "  p25:       " << get_percentile(unique_labelsets_values, 25) << "\n";
    //   std::cout << "  Median:    " << get_percentile(unique_labelsets_values, 50) << "\n";
    //   std::cout << "  p75:       " << get_percentile(unique_labelsets_values, 75) << "\n";
    //   std::cout << "  Max (p100):" << unique_labelsets_values.back() << "\n";
      
    //   std::cout << "\nRecall Distribution Among Low Recall Queries:\n";
    //   std::cout << "  Min (p0):  " << get_percentile_float(recall_values, 0) << "\n";
    //   std::cout << "  p25:       " << get_percentile_float(recall_values, 25) << "\n";
    //   std::cout << "  Median:    " << get_percentile_float(recall_values, 50) << "\n";
    //   std::cout << "  p75:       " << get_percentile_float(recall_values, 75) << "\n";
    //   std::cout << "  Max (p100):" << get_percentile_float(recall_values, 100) << "\n";
      
    //   std::cout << "========================================\n\n";
    // }
  }

  if (base_vecs) {
    delete[] base_vecs;
  }
  delete[] query_vecs;
  return 0;
}
