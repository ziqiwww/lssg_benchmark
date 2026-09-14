#include "bench_utils.hh"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " --query_vec <file> --query_labelset <file> --base_vec <file> --base_labelset <file> --output_prefix <prefix> [--min_pass <int>] [--scenario <containment|equality|overlap>]" << std::endl;
        return 1;
    }
    std::string query_vec_file, query_label_file, base_vec_file, base_label_file, output_prefix, scenario_str = "containment";
    int min_pass = 4000;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--query_vec") == 0) {
            query_vec_file = argv[++i];
        } else if (strcmp(argv[i], "--query_labelset") == 0) {
            query_label_file = argv[++i];
        } else if (strcmp(argv[i], "--base_vec") == 0) {
            base_vec_file = argv[++i];
        } else if (strcmp(argv[i], "--base_labelset") == 0) {
            base_label_file = argv[++i];
        } else if (strcmp(argv[i], "--output_prefix") == 0) {
            output_prefix = argv[++i];
        } else if (strcmp(argv[i], "--min_pass") == 0) {
            min_pass = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--scenario") == 0) {
            scenario_str = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            return 1;
        }
    }
    if (query_vec_file.empty() || query_label_file.empty() || base_vec_file.empty() || base_label_file.empty() || output_prefix.empty()) {
        std::cerr << "Missing required arguments." << std::endl;
        return 1;
    }
    // Load base vectors
    size_t base_dim = 0, nb_vec = 0;
    float* base_vecs_raw = benchmark::fvecs_read(base_vec_file, base_dim, nb_vec);
    std::vector<std::vector<float>> base_vecs(nb_vec, std::vector<float>(base_dim));
    for (size_t i = 0; i < nb_vec; ++i) {
        for (size_t j = 0; j < base_dim; ++j) {
            base_vecs[i][j] = base_vecs_raw[i * base_dim + j];
        }
    }
    delete[] base_vecs_raw;
    // Load query vectors
    size_t query_dim = 0, nq_raw = 0;
    float* query_vecs_raw = benchmark::fvecs_read(query_vec_file, query_dim, nq_raw);
    std::vector<std::vector<float>> query_vecs(nq_raw, std::vector<float>(query_dim));
    for (size_t i = 0; i < nq_raw; ++i) {
        for (size_t j = 0; j < query_dim; ++j) {
            query_vecs[i][j] = query_vecs_raw[i * query_dim + j];
        }
    }
    delete[] query_vecs_raw;
    // Load query labelsets
    auto query_labelsets = benchmark::LoadLabelSets(query_label_file);
    // Load base labelsets
    auto base_labelsets = benchmark::LoadLabelSets(base_label_file);
    size_t nb = base_labelsets.size();
    if (nb_vec != nb) {
        std::cerr << "Base vector count (" << nb_vec << ") does not match base labelset count (" << nb << ")." << std::endl;
        return 1;
    }
    if (query_dim != base_dim) {
        std::cerr << "Query vector dimension (" << query_dim << ") does not match base vector dimension (" << base_dim << ")." << std::endl;
        return 1;
    }
    size_t nq = query_vecs.size();
    size_t nql = query_labelsets.size();
    // Enlarge query vectors if necessary
    // if (nq < nql) {
    //     std::mt19937 gen(42); // fixed seed for reproducibility
    //     std::uniform_int_distribution<size_t> dist(0, nb_vec - 1);
    //     for (size_t i = nq; i < nql; ++i) {
    //         size_t idx = dist(gen);
    //         query_vecs.push_back(base_vecs[idx]);
    //     }
    //     nq = nql;
    //     std::cout << "Enlarged query vectors from " << nq_raw << " to " << nq << " by sampling from base vectors." << std::endl;
    // } else if (nql < nq) {
    //     query_labelsets.resize(nq);
    //     std::cout << "Resized query labelsets to match query vectors: " << nq << std::endl;
    // }
    // if (nq == 0 || nb == 0) {
    //     std::cerr << "No queries or base labelsets loaded." << std::endl;
    //     return 1;
    // }
    // Parse scenario
    lssg::LabelFilterConfig::Type scenario;
    if (scenario_str == "containment") scenario = lssg::LabelFilterConfig::LABEL_CONTAINMENT;
    else if (scenario_str == "equality") scenario = lssg::LabelFilterConfig::LABEL_EQUALITY;
    else if (scenario_str == "overlap") scenario = lssg::LabelFilterConfig::LABEL_OVERLAP;
    else {
        std::cerr << "Unknown scenario: " << scenario_str << std::endl;
        return 1;
    }
    // Filter queries (avoid data races by collecting flags in parallel, then compact)
    std::vector<char> keep_flags(nq, 0);
    #pragma omp parallel for
    for (size_t iq = 0; iq < nq; ++iq) {
        size_t passing_count = 0;
        for (size_t ib = 0; ib < nb; ++ib) {
            if (benchmark::LabelFilterMatch(base_labelsets[ib], query_labelsets[iq], scenario)) {
                passing_count++;
            }
        }
        if (passing_count >= static_cast<size_t>(min_pass)) {
            keep_flags[iq] = 1;
        }
    }
    std::vector<size_t> kept_indices;
    kept_indices.reserve(nq);
    for (size_t iq = 0; iq < nq; ++iq) {
        if (keep_flags[iq]) kept_indices.push_back(iq);
    }
    std::cout << "Kept " << kept_indices.size() << " / " << nq << " queries with passing_count >= " << min_pass << std::endl;
    // Write filtered query vectors
    std::ofstream ofs_vec(output_prefix + "_query.fvecs", std::ios::binary);
    for (size_t idx : kept_indices) {
        int d = static_cast<int>(query_dim);
        ofs_vec.write(reinterpret_cast<const char*>(&d), sizeof(int));
        ofs_vec.write(reinterpret_cast<const char*>(query_vecs[idx].data()), query_dim * sizeof(float));
    }
    ofs_vec.close();
    // Write filtered query vectors in .bin format
    std::ofstream ofs_bin(output_prefix + "_query.bin", std::ios::binary);
    int n_kept = static_cast<int>(kept_indices.size());
    ofs_bin.write(reinterpret_cast<const char*>(&n_kept), sizeof(int));
    int d = static_cast<int>(query_dim);
    ofs_bin.write(reinterpret_cast<const char*>(&d), sizeof(int));
    for (size_t idx : kept_indices) {
        ofs_bin.write(reinterpret_cast<const char*>(query_vecs[idx].data()), query_dim * sizeof(float));
    }
    ofs_bin.close();
    // Write filtered query labelsets
    std::ofstream ofs_label(output_prefix + "_query.txt");
    for (size_t idx : kept_indices) {
        const auto& labels = query_labelsets[idx];
        for (size_t i = 0; i < labels.size(); ++i) {
            if (i > 0) ofs_label << ",";
            ofs_label << labels[i];
        }
        ofs_label << "\n";
    }
    ofs_label.close();
    std::cout << "Wrote filtered query vectors and labelsets to " << output_prefix << "_query.fvecs, " << output_prefix << "_query.bin, and " << output_prefix << "_query.txt" << std::endl;
    return 0;
}
