#!/bin/bash

# Partition queries by quantiles (equal-sized groups sorted by specificity)
# This script runs the partition_by_quantile program for different datasets and scenarios

# ============================================================================
# CONFIGURATION - Uncomment the dataset you want to process
# ============================================================================

# sift1m
db="sift1m"
num_label=12
base_vec=~/work/data/vecs/sift1m/sift_base.fvecs
query_vec=~/work/data/vecs/sift1m/sift_query.fvecs
base_labelset=./attrdata/label/sift1m_base_n1000000_l${num_label}_zipf.txt
containment_query_labelset=./attrdata/label/sift1m_query_n10000_l${num_label}_zipf_containment.txt
equality_query_labelset=./attrdata/label/sift1m_query_n10000_l${num_label}_zipf_equality.txt
overlap_query_labelset=./attrdata/label/sift1m_query_n10000_l${num_label}_zipf_overlap.txt


# # GIST1M
# db="gist1m"
# num_label=12
# base_vec=~/work/data/vecs/gist1m/gist_base.fvecs
# query_vec=~/work/data/vecs/gist1m/gist_query.fvecs
# base_labelset=./attrdata/label/gist1m_base_n1000000_l${num_label}_zipf.txt
# containment_query_labelset=./attrdata/label/gist1m_query_n10000_l${num_label}_zipf_containment.txt
# equality_query_labelset=./attrdata/label/gist1m_query_n10000_l${num_label}_zipf_equality.txt
# overlap_query_labelset=./attrdata/label/gist1m_query_n10000_l${num_label}_zipf_overlap.txt

# LAION1M
# db="LAION1M"
# num_label=30
# base_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_base.fvecs
# query_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_query.fvecs
# base_labelset=./attrdata/label/LAION1M_base_n1000448_l${num_label}_zipf.txt
# containment_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_zipf_containment.txt
# equality_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_zipf_equality.txt
# overlap_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_zipf_overlap.txt

# # TRIPCLICK
# db="tripclick"
# num_label=29
# base_vec=~/work/data/filterbenchmark/tripclick/tripclick_base.fvecs
# query_vec=~/work/data/filterbenchmark/tripclick/tripclick_query_and.fvecs
# base_labelset=~/work/data/filterbenchmark/tripclick/label_base.txt
# containment_query_labelset=./attrdata/label/tripclick_query_n10000_l${num_label}_zipf_containment.txt
# equality_query_labelset=./attrdata/label/tripclick_query_n10000_l${num_label}_zipf_equality.txt
# overlap_query_labelset=./attrdata/label/tripclick_query_n10000_l${num_label}_zipf_overlap.txt

# YFCC
# db="yfcc"
# num_label=65521
# base_vec=~/work/data/filterbenchmark/yfcc/yfcc_base.fvecs
# query_vec=~/work/data/filterbenchmark/yfcc/yfcc_query_and.fvecs
# base_labelset=~/work/data/filterbenchmark/yfcc/label_base.txt
# containment_query_labelset=~/work/lssg/data/filterbenchmark/yfcc/yfcc_query_and.txt
# equality_query_labelset=~/work/lssg/data/filterbenchmark/yfcc/yfcc_query_equal.txt
# overlap_query_labelset=~/work/lssg/data/filterbenchmark/yfcc/yfcc_query_or.txt

# YTB_AUDIO
# db="ytb_audio"
# num_label=3862
# base_vec=~/work/data/filterbenchmark/ytb_audio/ytb_audio_base.fvecs
# query_vec=~/work/data/filterbenchmark/ytb_audio/ytb_audio_query_and.fvecs
# base_labelset=~/work/data/filterbenchmark/ytb_audio/label_base.txt
# containment_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_audio/ytb_audio_query_and.txt
# equality_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_audio/ytb_audio_query_equal.txt
# overlap_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_audio/ytb_audio_query_or.txt

# YTB_VIDEO
# db="ytb_video"
# num_label=3862
# base_vec=~/work/data/filterbenchmark/ytb_video/ytb_video_base.fvecs
# query_vec=~/work/data/filterbenchmark/ytb_video/ytb_video_query_and.fvecs
# base_labelset=~/work/data/filterbenchmark/ytb_video/label_base.txt
# containment_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_video/ytb_video_query_and.txt
# equality_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_video/ytb_video_query_equal.txt
# overlap_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_video/ytb_video_query_or.txt

# # YFCC-10M
# db="yfcc-10m"
# num_label=200386
# base_vec=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/base.10M.fvecs
# query_vec=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/query.10K.fvecs
# base_labelset=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/label_base_filled.txt
# containment_query_labelset=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/label_query_filled.txt
# equality_query_labelset=/root/work/lssg/example/attrdata/label/yfcc-10m_query_n10000_l200386_zipf_equality.txt
# overlap_query_labelset=/root/work/lssg/example/attrdata/label/yfcc-10m_query_n10000_l200386_zipf_overlap.txt

# # DEEP-10M
# db="deep-10m"
# num_label=65535
# base_vec=/root/work/lssg/data/vecs/deep1b/deep10m_base.fvecs
# query_vec=/root/work/lssg/data/vecs/deep1b/deep1B_queries.fvecs
# base_labelset=/root/work/lssg/example/attrdata/label/deep-10m_base_n10000000_l65535_zipf.txt
# containment_query_labelset=/root/work/lssg/example/attrdata/label/deep-10m_query_n10000_l65535_zipf_containment.txt
# equality_query_labelset=/root/work/lssg/example/attrdata/label/deep-10m_query_n10000_l65535_zipf_equality.txt
# overlap_query_labelset=/root/work/lssg/example/attrdata/label/deep-10m_query_n10000_l65535_zipf_overlap.txt

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

# Build the program if needed
echo "Building partition_by_quantile..."
cd ../build
cmake .. && make -j8 partition_by_quantile
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi
cd ../example

# Process each scenario
for scenario in containment equality overlap; do
    echo ""
    echo "================================================================================"
    echo "Processing ${db} - ${scenario}"
    echo "================================================================================"
    
    # Set query labelset based on scenario
    if [ "$scenario" = "containment" ]; then
        query_labelset=$containment_query_labelset
    elif [ "$scenario" = "equality" ]; then
        query_labelset=$equality_query_labelset
    elif [ "$scenario" = "overlap" ]; then
        query_labelset=$overlap_query_labelset
    fi
    
    # Check if query labelset exists
    if [ ! -f "$query_labelset" ]; then
        echo "Warning: Query labelset not found: $query_labelset"
        echo "Skipping ${scenario}..."
        continue
    fi
    
    # Set output directory
    output_dir=./quantile_partitions/${db}_${scenario}
    
    # Run partition program
    ../build/bin/partition_by_quantile \
        --base_vec $base_vec \
        --query_vec $query_vec \
        --base_labelset $base_labelset \
        --query_labelset $query_labelset \
        --output_dir $output_dir \
        --dataset_name $db \
        --scenario $scenario
    
    if [ $? -eq 0 ]; then
        echo "Successfully partitioned ${db} - ${scenario}"
    else
        echo "Error partitioning ${db} - ${scenario}"
    fi
done

echo ""
echo "================================================================================"
echo "All quantile partitioning completed!"
echo "================================================================================"
