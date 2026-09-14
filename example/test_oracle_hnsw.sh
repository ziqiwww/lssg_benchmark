#!/bin/bash

# Oracle HNSW Test Script
# Runs oracle HNSW tests for different datasets and scenarios
# This script aligns with dataset configurations from search_polabel.sh
#
# NOTE: The implementation now groups queries by label set and processes them
# group-by-group, clearing indices after each group to minimize memory usage.
# This eliminates the need for frequent disk I/O during testing.

set -e

# Index storage directory
# Indices are saved to disk for potential reuse across runs
index_dir="./index/oracle_indices"
# max_memory_indices is kept for backward compatibility but has minimal effect
# since indices are cleared after each query group
max_memory_indices=50

# Dataset configurations (matching search_polabel.sh)
# Uncomment the dataset you want to test

# db="sift1m"
# m=16
# efc=128
# k=10
# num_label=12
# space="l2"
# base_vec="../data/vecs/sift1m/sift_base.fvecs"
# query_vec="../data/vecs/sift1m/sift_query.fvecs"
# base_labelset="./attrdata/label/sift1m_base_n1000000_l${num_label}_zipf.txt"
# containment_query_labelset="./attrdata/label/sift1m_query_n10000_l${num_label}_zipf_containment.txt"
# equality_query_labelset="./attrdata/label/sift1m_query_n10000_l${num_label}_zipf_equality.txt"
# overlap_query_labelset="./attrdata/label/sift1m_query_n10000_l${num_label}_zipf_overlap.txt"

# db="gist1m"
# m=16
# efc=128
# k=10
# num_label=12
# space="l2"
# base_vec="../data/vecs/gist1m/gist_base.fvecs"
# query_vec="../data/vecs/gist1m/gist_query.fvecs"
# base_labelset="./attrdata/label/gist1m_base_n1000000_l${num_label}_zipf.txt"
# containment_query_labelset="./attrdata/label/gist1m_query_n10000_l${num_label}_zipf_containment.txt"
# equality_query_labelset="./attrdata/label/gist1m_query_n10000_l${num_label}_zipf_equality.txt"
# overlap_query_labelset="./attrdata/label/gist1m_query_n10000_l${num_label}_zipf_overlap.txt"

# db="LAION1M"
# m=16
# efc=128
# k=10
# num_label=30
# space="l2"
# base_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_base.fvecs
# query_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_query.fvecs
# base_labelset=./attrdata/label/LAION1M_base_n1000448_l${num_label}_zipf.txt
# containment_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_zipf_containment.txt
# equality_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_zipf_equality.txt
# overlap_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_zipf_overlap.txt

# db="tripclick"
# m=16
# efc=128
# k=10
# num_label=29
# space="l2"
# base_vec=~/work/data/filterbenchmark/tripclick/tripclick_base.fvecs
# query_vec=~/work/data/filterbenchmark/tripclick/tripclick_query_and.fvecs
# base_labelset=~/work/data/filterbenchmark/tripclick/label_base.txt
# containment_query_labelset=./attrdata/label/tripclick_query_n10000_l${num_label}_zipf_containment.txt
# equality_query_labelset=./attrdata/label/tripclick_query_n10000_l${num_label}_zipf_equality.txt
# overlap_query_labelset=./attrdata/label/tripclick_query_n10000_l${num_label}_zipf_overlap.txt

# db="ytb_audio"
# m=16
# efc=128
# k=10
# num_label=3862
# space="l2"
# base_vec=~/work/data/filterbenchmark/ytb_audio/ytb_audio_base.fvecs
# query_vec=~/work/data/filterbenchmark/ytb_audio/ytb_audio_query_and.fvecs
# base_labelset=~/work/data/filterbenchmark/ytb_audio/label_base.txt
# containment_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_audio/ytb_audio_query_and.txt
# equality_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_audio/ytb_audio_query_equal.txt
# overlap_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_audio/ytb_audio_query_or.txt

# db="ytb_video"
# m=24
# efc=128
# k=10
# num_label=3862
# space="l2"
# base_vec=~/work/data/filterbenchmark/ytb_video/ytb_video_base.fvecs
# query_vec=~/work/data/filterbenchmark/ytb_video/ytb_video_query_and.fvecs
# base_labelset=~/work/data/filterbenchmark/ytb_video/label_base.txt
# containment_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_video/ytb_video_query_and.txt
# equality_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_video/ytb_video_query_equal.txt
# overlap_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_video/ytb_video_query_or.txt

# db="yfcc"
# m=16
# efc=128
# k=10
# num_label=65521
# space="l2"
# base_vec=~/work/data/filterbenchmark/yfcc/yfcc_base.fvecs
# query_vec=~/work/data/filterbenchmark/yfcc/yfcc_query_and.fvecs
# base_labelset=~/work/data/filterbenchmark/yfcc/label_base.txt
# containment_query_labelset=~/work/lssg/data/filterbenchmark/yfcc/yfcc_query_and.txt
# equality_query_labelset=~/work/lssg/data/filterbenchmark/yfcc/yfcc_query_equal.txt
# overlap_query_labelset=~/work/lssg/data/filterbenchmark/yfcc/yfcc_query_or.txt

# Ground truth directory
gt_dir="./gt/gt_label/${db}_k${k}"
gt_containment="$gt_dir/containment_l${num_label}.bin"
gt_equality="$gt_dir/equality_l${num_label}.bin"
gt_overlap="$gt_dir/overlap_l${num_label}.bin"
mkdir -p "$gt_dir"

# Create index directory
mkdir -p "$index_dir"

# Build test executable
cd ../build
cmake .. && make test_oracle_hnsw -j8
cd ../example

echo "=================================================="
echo "Oracle HNSW Test Configuration"
echo "=================================================="
echo "Dataset: $db"
echo "M: $m"
echo "ef_construction: $efc"
echo "k: $k"
echo "num_label: $num_label"
echo "space: $space"
echo "Index directory: $index_dir"
echo "Max memory indices: $max_memory_indices"
echo "=================================================="
echo ""

# Generate ground truth if not exists (using existing tools)
if [ ! -f "$gt_containment" ]; then
    echo "Generating containment ground truth"
    ../build/bin/gen_label_gt \
        --basevec $base_vec \
        --queryvec $query_vec \
        --base_labelset $base_labelset \
        --query_labelset $containment_query_labelset \
        --gt_file $gt_containment \
        --space $space \
        --type containment \
        --k $k
fi

if [ ! -f "$gt_equality" ]; then
    echo "Generating equality ground truth"
    ../build/bin/gen_label_gt \
        --basevec $base_vec \
        --queryvec $query_vec \
        --base_labelset $base_labelset \
        --query_labelset $equality_query_labelset \
        --gt_file $gt_equality \
        --space $space \
        --type equality \
        --k $k
fi

if [ ! -f "$gt_overlap" ]; then
    echo "Generating overlap ground truth"
    ../build/bin/gen_label_gt \
        --basevec $base_vec \
        --queryvec $query_vec \
        --base_labelset $base_labelset \
        --query_labelset $overlap_query_labelset \
        --gt_file $gt_overlap \
        --space $space \
        --type overlap \
        --k $k
fi

# Run containment scenario
echo "Running Oracle HNSW - Containment scenario..."
../build/bin/test_oracle_hnsw \
    $base_vec \
    $query_vec \
    $base_labelset \
    $containment_query_labelset \
    containment \
    $gt_containment \
    $space \
    $m \
    $efc \
    $k \
    $index_dir \
    $max_memory_indices \
    > oracle_containment.csv

# Run equality scenario
echo "Running Oracle HNSW - Equality scenario..."
../build/bin/test_oracle_hnsw \
    $base_vec \
    $query_vec \
    $base_labelset \
    $equality_query_labelset \
    equality \
    $gt_equality \
    $space \
    $m \
    $efc \
    $k \
    $index_dir \
    $max_memory_indices \
    > oracle_equality.csv

# Run overlap scenario
echo "Running Oracle HNSW - Overlap scenario..."
../build/bin/test_oracle_hnsw \
    $base_vec \
    $query_vec \
    $base_labelset \
    $overlap_query_labelset \
    overlap \
    $gt_overlap \
    $space \
    $m \
    $efc \
    $k \
    $index_dir \
    $max_memory_indices \
    > oracle_overlap.csv

echo ""
echo "=================================================="
echo "Oracle HNSW tests completed!"
echo "Results saved to:"
echo "  - oracle_containment.csv"
echo "  - oracle_equality.csv"
echo "  - oracle_overlap.csv"
echo "=================================================="
