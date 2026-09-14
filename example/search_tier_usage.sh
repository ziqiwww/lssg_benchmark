#!/bin/bash

# Tier-usage experiment for PoIndex
# Reports per-layer visited vertices and candidate contributions
# for each filter type and selectivity bucket
#
# Datasets: LAION1M and ytb_audio

set -e

k=10
tier_usage_ef=1000

# ============================================================
# LAION1M Configuration
# ============================================================
run_laion() {
    db="LAION1M"
    m=16
    efc=128
    num_label=30
    space="l2"
    distribution="zipf"
    base_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_base.fvecs
    query_vec=/root/work/data/filterbenchmark/LAION1M/LAION1M_query.fvecs
    base_labelset=./attrdata/label/LAION1M_base_n1000448_l${num_label}_${distribution}.txt
    containment_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_${distribution}_containment.txt
    equality_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_${distribution}_equality.txt
    overlap_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_${distribution}_overlap.txt

    index=./index/${db}_${m}_${efc}_${space}_label${num_label}.poi
    scope=./index/${db}_label${num_label}.sco

    run_tier_usage "$db" "$base_vec" "$base_labelset" "$space" "$index" "$scope" \
        "$containment_query_labelset" "$equality_query_labelset" "$overlap_query_labelset" \
        "$query_vec" "$query_vec" "$query_vec_containment" "$query_vec_overlap"
}

# ============================================================
# ytb_audio Configuration
# ============================================================
run_ytb_audio() {
    db="ytb_audio"
    m=16
    efc=128
    num_label=3862
    space="l2"
    base_vec=~/work/data/filterbenchmark/ytb_audio/ytb_audio_base.fvecs
    query_vec=~/work/data/filterbenchmark/ytb_audio/ytb_audio_query_and.fvecs
    base_labelset=~/work/data/filterbenchmark/ytb_audio/label_base.txt
    containment_query_labelset=/root/work/wow/data/filterbenchmark/ytb_audio/ytb_audio_query_and.txt
    equality_query_labelset=/root/work/wow/data/filterbenchmark/ytb_audio/ytb_audio_query_equal.txt
    overlap_query_labelset=/root/work/wow/data/filterbenchmark/ytb_audio/ytb_audio_query_or.txt

    index=./index/${db}_${m}_${efc}_${space}_label${num_label}.poi
    scope=./index/${db}_label${num_label}.sco

    run_tier_usage "$db" "$base_vec" "$base_labelset" "$space" "$index" "$scope" \
        "$containment_query_labelset" "$equality_query_labelset" "$overlap_query_labelset" \
        "$query_vec" "$query_vec" "" ""
}

# ============================================================
# Common tier-usage runner
# ============================================================
run_tier_usage() {
    local db=$1 base_vec=$2 base_labelset=$3 space=$4 index=$5 scope=$6
    local containment_labelset=$7 equality_labelset=$8 overlap_labelset=$9
    local query_vec=${10} query_vec_equality=${11}
    local query_vec_containment=${12} query_vec_overlap=${13}

    local output_dir="./tier_usage_results"
    mkdir -p "$output_dir"

    # Use specialized query vectors if available, otherwise use default
    local qvec_containment="${query_vec_containment:-$query_vec}"
    local qvec_overlap="${query_vec_overlap:-$query_vec}"
    local qvec_equality="${query_vec_equality:-$query_vec}"

    # Run containment scenario
    echo "=== [$db] Tier usage: containment (ef=$tier_usage_ef) ==="
    ../build/bin/search_po_label \
        --query_vec "$qvec_containment" \
        --query_labelset "$containment_labelset" \
        --basevec "$base_vec" \
        --base_labelset "$base_labelset" \
        --index_location "$index" \
        --scope_location "$scope" \
        --space "$space" \
        --k "$k" \
        --tier_usage_csv "$output_dir/tier_usage_${db}_containment.csv" \
        --tier_usage_ef "$tier_usage_ef" \
        --tier_filter_type containment \
        > /dev/null

    # Run equality scenario
    echo "=== [$db] Tier usage: equality (ef=$tier_usage_ef) ==="
    ../build/bin/search_po_label \
        --query_vec "$qvec_equality" \
        --query_labelset "$equality_labelset" \
        --basevec "$base_vec" \
        --base_labelset "$base_labelset" \
        --index_location "$index" \
        --scope_location "$scope" \
        --space "$space" \
        --k "$k" \
        --tier_usage_csv "$output_dir/tier_usage_${db}_equality.csv" \
        --tier_usage_ef "$tier_usage_ef" \
        --tier_filter_type equality \
        > /dev/null

    # Run overlap scenario
    echo "=== [$db] Tier usage: overlap (ef=$tier_usage_ef) ==="
    ../build/bin/search_po_label \
        --query_vec "$qvec_overlap" \
        --query_labelset "$overlap_labelset" \
        --basevec "$base_vec" \
        --base_labelset "$base_labelset" \
        --index_location "$index" \
        --scope_location "$scope" \
        --space "$space" \
        --k "$k" \
        --tier_usage_csv "$output_dir/tier_usage_${db}_overlap.csv" \
        --tier_usage_ef "$tier_usage_ef" \
        --tier_filter_type overlap \
        > /dev/null

    echo "=== [$db] Tier usage complete ==="
}

# ============================================================
# Main
# ============================================================
echo "Building..."
cd ../build
cmake .. && make -j8 search_po_label
cd ../example

echo ""
echo "=============================================="
echo "  Tier-Usage Experiment"
echo "  Datasets: LAION1M, ytb_audio"
echo "  EF for tier stats: $tier_usage_ef"
echo "=============================================="
echo ""

run_laion
run_ytb_audio

echo ""
echo "All tier-usage experiments completed."
echo "Results in ./tier_usage_results/"
ls -la ./tier_usage_results/
