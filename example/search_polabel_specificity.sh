#!/bin/bash

# Search queries partitioned by specificity levels
# This script runs searches on queries grouped by their specificity ranges
# and generates performance results for each specificity level

# ============================================================================
# CONFIGURATION - Uncomment the dataset you want to process
# ============================================================================

# # GIST1M
db="gist1m"
m=16
efc=128
k=10
num_label=12
space="l2"
base_vec=~/work/data/vecs/gist1m/gist_base.fvecs
base_labelset=./attrdata/label/gist1m_base_n1000000_l${num_label}_zipf.txt

# LAION1M
# db="LAION1M"
# m=16
# efc=128
# k=10
# num_label=30
# space="l2"
# base_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_base.fvecs
# base_labelset=./attrdata/label/LAION1M_base_n1000448_l${num_label}_zipf.txt

# YFCC
# db="yfcc"
# m=16
# efc=128
# k=10
# num_label=65521
# space="l2"
# base_vec=~/work/data/filterbenchmark/yfcc/yfcc_base.fvecs
# base_labelset=~/work/data/filterbenchmark/yfcc/label_base.txt

# # YFCC-10M
# db="yfcc-10m"
# m=16
# efc=128
# k=10
# num_label=200386
# space="l2"
# base_vec=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/base.10M.fvecs
# base_labelset=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/label_base_filled.txt

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

# Index paths (shared across all specificity levels)
index=./index/${db}_${m}_${efc}_${space}_label${num_label}.poi
scope=./index/${db}_label${num_label}.sco

# Specificity bins
specificities=("0-20" "20-40" "40-60" "60-80" "80-100")

# Build the search program if needed
echo "Building search_po_label..."
cd ../build
cmake .. && make -j8 search_po_label gen_label_gt
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi
cd ../example

# Process each scenario
for scenario in containment overlap; do
    echo ""
    echo "================================================================================"
    echo "Processing ${db} - ${scenario}"
    echo "================================================================================"
    
    partition_dir=./specificity_partitions/${db}_${scenario}
    
    # Check if partition directory exists
    if [ ! -d "$partition_dir" ]; then
        echo "Warning: Partition directory not found: $partition_dir"
        echo "Please run partition_specificity.sh first!"
        continue
    fi
    
    # Process each specificity level
    for spec in "${specificities[@]}"; do
        echo ""
        echo "--------------------------------------------------------------------------------"
        echo "Specificity: ${spec}%"
        echo "--------------------------------------------------------------------------------"
        
        # Check if files exist for this specificity level
        query_vec="${partition_dir}/${db}_${scenario}_specificity${spec}_query.fvecs"
        query_labelset="${partition_dir}/${db}_${scenario}_specificity${spec}_query.txt"
        
        if [ ! -f "$query_vec" ]; then
            echo "  Skipping ${spec}% (no queries in this range)"
            continue
        fi
        
        # Ground truth path
        gt_dir="./gt/gt_label_specificity/${db}_k${k}/${scenario}"
        gt_file="$gt_dir/specificity${spec}_l${num_label}.bin"
        mkdir -p "$gt_dir"
        
        # Generate ground truth if it doesn't exist
        if [ ! -f "$gt_file" ]; then
            echo "  Generating ground truth for specificity ${spec}%..."
            ../build/bin/gen_label_gt \
                --basevec $base_vec \
                --queryvec $query_vec \
                --base_labelset $base_labelset \
                --query_labelset $query_labelset \
                --gt_file $gt_file \
                --space $space \
                --type $scenario \
                --k $k
            
            if [ $? -ne 0 ]; then
                echo "  Error generating ground truth for ${spec}%"
                continue
            fi
        else
            echo "  Ground truth exists: $gt_file"
        fi
        
        # Output CSV file
        output_dir="./results_specificity/${db}_${scenario}"
        mkdir -p "$output_dir"
        output_csv="${output_dir}/poindex_${scenario}_specificity${spec}.csv"
        
        # Run search
        echo "  Running search for specificity ${spec}%..."
        
        # Construct GT argument based on scenario
        if [ "$scenario" = "containment" ]; then
            gt_arg="--gt_containment $gt_file"
        elif [ "$scenario" = "equality" ]; then
            gt_arg="--gt_equality $gt_file"
        elif [ "$scenario" = "overlap" ]; then
            gt_arg="--gt_overlap $gt_file"
        fi
        
        ../build/bin/search_po_label \
            --query_vec $query_vec \
            --query_labelset $query_labelset \
            --basevec $base_vec \
            --base_labelset $base_labelset \
            --index_location $index \
            --scope_location $scope \
            --space $space \
            --k $k \
            $gt_arg \
            > $output_csv
        
        if [ $? -eq 0 ]; then
            echo "  Successfully searched specificity ${spec}%"
            echo "  Results saved to: $output_csv"
            
            # Print a summary of the results
            if [ -f "$output_csv" ]; then
                echo "  Summary:"
                tail -5 "$output_csv" | head -3
            fi
        else
            echo "  Error searching specificity ${spec}%"
        fi
    done
    
    echo ""
    echo "Completed ${db} - ${scenario}"
done

echo ""
echo "================================================================================"
echo "All searches completed!"
echo "================================================================================"
echo ""
echo "Results are saved in: ./results_specificity/${db}_*/"
echo ""
echo "To analyze results, you can:"
echo "  1. Check individual CSV files for detailed metrics"
echo "  2. Compare performance across specificity levels"
echo "  3. Plot QPS vs Recall curves for each specificity range"
