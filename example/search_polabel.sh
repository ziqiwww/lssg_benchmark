#!/bin/bash

# db="sift1m"
# m=16
# efc=128
# k=10
# num_label=12
# space="l2"
# distribution="zipf" # zipf multi_normial poisson
# base_vec="../data/vecs/sift1m/sift_base.fvecs"
# query_vec="../data/vecs/sift1m/sift_query.fvecs"
# base_labelset="./attrdata/label/sift1m_base_n1000000_l${num_label}_${distribution}.txt"
# containment_query_labelset="./attrdata/label/sift1m_query_n10000_l${num_label}_${distribution}_containment.txt"
# equality_query_labelset="./attrdata/label/sift1m_query_n10000_l${num_label}_${distribution}_equality.txt"
# overlap_query_labelset="./attrdata/label/sift1m_query_n10000_l${num_label}_${distribution}_overlap.txt"

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
# distribution="zipf" # zipf multi_normial poisson uniform
# base_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_base.fvecs
# query_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_query.fvecs
# # base_labelset=./attrdata/label/LAION1M_base_n1000448_l${num_label}_${distribution}.txt
# # containment_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_${distribution}_containment.txt
# # equality_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_${distribution}_equality.txt
# # overlap_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_${distribution}_overlap.txt
# base_labelset=./attrdata/label/LAION1M_base_n1000448_l${num_label}_${distribution}.txt
# containment_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_${distribution}_containment.txt
# equality_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_${distribution}_equality.txt
# overlap_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_${distribution}_overlap.txt
# index_type="minhash"
# index_type="ivf"

# multi label
# db="LAION1M"
# m=16
# efc=128
# k=10
# num_label=256
# space="l2"
# distribution="zipf" # zipf multi_normial poisson uniform
# base_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_base.fvecs
# base_labelset=./attrdata/label/LAION1M_base_n1000448_l${num_label}_${distribution}.txt
# query_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_query.fvecs
# query_vec_containment=/root/work/data/filterbenchmark/LAION1M/LAION1M_query_l${num_label}_pass4096_containment_query.fvecs
# query_vec_overlap=/root/work/data/filterbenchmark/LAION1M/LAION1M_query_l${num_label}_pass4096_overlap_query.fvecs
# containment_query_labelset=/root/work/data/filterbenchmark/LAION1M/LAION1M_query_l${num_label}_pass4096_containment_query.txt
# equality_query_labelset=./attrdata/label/LAION1M_query_n10000_l${num_label}_${distribution}_equality.txt
# overlap_query_labelset=/root/work/data/filterbenchmark/LAION1M/LAION1M_query_l${num_label}_pass4096_overlap_query.txt
# index_type="minhash"
# # index_type="ivf"

# db="tripclick"
# m=16
# efc=128
# k=10
# num_label=29
# space="l2"
# distribution="zipf" # zipf multi_normial poisson
# base_vec=~/work/data/filterbenchmark/tripclick/tripclick_base.fvecs
# query_vec=~/work/data/filterbenchmark/tripclick/tripclick_query_and.fvecs
# base_labelset=~/work/data/filterbenchmark/tripclick/label_base.txt
# # base_labelset=/root/work/lssg/example/attrdata/label/tripclick_base_n1055976_l29_${distribution}.txt
# containment_query_labelset=./attrdata/label/tripclick_query_n10000_l${num_label}_${distribution}_containment.txt
# equality_query_labelset=./attrdata/label/tripclick_query_n10000_l${num_label}_${distribution}_equality.txt
# overlap_query_labelset=./attrdata/label/tripclick_query_n10000_l${num_label}_${distribution}_overlap.txt

db="ytb_audio"
m=16
efc=128
k=10
num_label=3862
space="l2"
base_vec=~/work/data/filterbenchmark/ytb_audio/ytb_audio_base.fvecs
query_vec=~/work/data/filterbenchmark/ytb_audio/ytb_audio_query_and.fvecs
base_labelset=~/work/data/filterbenchmark/ytb_audio/label_base.txt
containment_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_audio/ytb_audio_query_and.txt
equality_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_audio/ytb_audio_query_equal.txt
overlap_query_labelset=/root/work/lssg/data/filterbenchmark/ytb_audio/ytb_audio_query_or.txt
# index_type="minhash"
index_type="ivf"


# db="ytb_video"
# m=16
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
# index_type="minhash"

# db="glove"
# m=16
# efc=200
# k=10
# num_label=2000
# space="l2"
# base_vec=~/work/data/vecs/glove-100/glove-100_base.fvecs
# query_vec=~/work/data/vecs/glove-100/glove-100_query.fvecs
# base_labelset=./attrdata/label/glove_base_n1183514_l${num_label}_zipf.txt
# containment_query_labelset=./attrdata/label/glove_query_n10000_l${num_label}_zipf_containment.txt
# equality_query_labelset=./attrdata/label/glove_query_n10000_l${num_label}_zipf_equality.txt
# overlap_query_labelset=./attrdata/label/glove_query_n10000_l${num_label}_zipf_overlap.txt

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
# index_type="minhash"
# # index_type="ivf"

# db="yfcc-10m"
# m=16
# efc=128
# k=10
# num_label=200386
# space="l2"
# base_vec=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/base.10M.fvecs
# query_vec=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/query.10K.fvecs
# # query_vec=/root/work/data/filterbenchmark/yfcc-10m/yfcc-10m_query_l200386_pass4000_containment_query.fvecs
# base_labelset=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/label_base_filled.txt
# # containment_query_labelset=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/label_query_filled.txt
# containment_query_labelset=/root/work/lssg/example/attrdata/label/yfcc-10m_query_n10000_l200386_zipf_containment.txt
# # equality_query_labelset=/root/work/lssg/example/attrdata/label/yfcc-10m_query_n10000_l200386_zipf_equality.txt
# # overlap_query_labelset=/root/work/lssg/example/attrdata/label/yfcc-10m_query_n10000_l200386_zipf_overlap.txt
# index_type="minhash"
# # index_type="ivf"

# db="wikipedia-5m"
# m=16
# efc=128
# k=10
# num_label=237445
# space="l2"
# base_vec=/root/work/data/vecs/wikipedia/wikipedia_5m_base.fvecs
# query_vec=/root/work/data/vecs/wikipedia/wikipedia_1k_query.fvecs
# base_labelset=/root/work/data/vecs/wikipedia/wikipedia_5m_base_labels.txt
# containment_query_labelset=/root/work/wow/example/attrdata/label/wikipedia-5m_query_n10000_l237445_zipf_containment.txt
# equality_query_labelset=/root/work/wow/example/attrdata/label/wikipedia-5m_query_n10000_l237445_zipf_equality.txt
# overlap_query_labelset=/root/work/wow/example/attrdata/label/wikipedia-5m_query_n10000_l237445_zipf_overlap.txt


# index=./index/${db}_${m}_${efc}_${space}_label${num_label}_${index_type}.poi
# scope=./index/${db}_label${num_label}_${index_type}.sco

index=./index/${db}_${m}_${efc}_${space}_label${num_label}.poi
scope=./index/${db}_label${num_label}.sco

gt_dir="./gt/gt_label/${db}_k${k}"
# gt_containment="$gt_dir/containment_l${num_label}_${distribution}.bin"
# gt_equality="$gt_dir/equality_l${num_label}_${distribution}.bin"
# gt_overlap="$gt_dir/overlap_l${num_label}_${distribution}.bin"

gt_containment="$gt_dir/containment_l${num_label}_${distribution}.bin"
gt_equality="$gt_dir/equality_l${num_label}_${distribution}.bin"
gt_overlap="$gt_dir/overlap_l${num_label}_${distribution}.bin"

mkdir -p "$(dirname $index)" "$gt_dir"

cd ../build
cmake .. && make -j8
cd ../example

query_vec_containment_to_use=$query_vec
query_vec_overlap_to_use=$query_vec
query_vec_equality_to_use=$query_vec

# if query vec specialized, use it
if [ -f "$query_vec_containment" ]; then
    query_vec_containment_to_use=$query_vec_containment
fi
if [ -f "$query_vec_overlap" ]; then
    query_vec_overlap_to_use=$query_vec_overlap
fi
if [ -f "$query_vec_equality" ]; then
    query_vec_equality_to_use=$query_vec_equality
fi

if [ ! -f "$gt_containment" ]; then
    echo "Generating containment ground truth"
    ../build/bin/gen_label_gt \
        --basevec $base_vec \
        --queryvec $query_vec_containment_to_use \
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
        --queryvec $query_vec_equality_to_use \
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
        --queryvec $query_vec_overlap_to_use \
        --base_labelset $base_labelset \
        --query_labelset $overlap_query_labelset \
        --gt_file $gt_overlap \
        --space $space \
        --type overlap \
        --k $k
fi

# Run containment scenario
echo "Running containment scenario..."
../build/bin/search_po_label \
    --query_vec $query_vec_containment_to_use \
    --query_labelset $containment_query_labelset \
    --basevec $base_vec \
    --base_labelset $base_labelset \
    --index_location $index \
    --scope_location $scope \
    --space $space \
    --k $k \
    --gt_containment $gt_containment \
    > poindex_containment.csv

# # Run equality scenario
echo "Running equality scenario..."
../build/bin/search_po_label \
    --query_vec $query_vec_equality_to_use \
    --query_labelset $equality_query_labelset \
    --basevec $base_vec \
    --base_labelset $base_labelset \
    --index_location $index \
    --scope_location $scope \
    --space $space \
    --k $k \
    --gt_equality $gt_equality \
    > poindex_equality.csv

# Run overlap scenario
echo "Running overlap scenario..."
../build/bin/search_po_label \
    --query_vec $query_vec_overlap_to_use \
    --query_labelset $overlap_query_labelset \
    --basevec $base_vec \
    --base_labelset $base_labelset \
    --index_location $index \
    --scope_location $scope \
    --space $space \
    --k $k \
    --gt_overlap $gt_overlap \
    > poindex_overlap.csv

echo "Search finished."

echo "All tests completed."
