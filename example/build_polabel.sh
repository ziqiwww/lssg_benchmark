#!/bin/bash

set -e

db="sift1m"
m=16
efc=128
num_label=12
base_vec="../data/vecs/sift1m/sift_base.fvecs"
base_labelset="./attrdata/label/sift1m_base_n1000000_l${num_label}_zipf.txt"
space="l2"

# db="gist1m"
# m=16
# efc=128
# num_label=12
# base_vec="../data/vecs/gist1m/gist_base.fvecs"
# base_labelset="./attrdata/label/gist1m_base_n1000000_l${num_label}_zipf.txt"
# space="l2"

# db="LAION1M"
# m=16
# efc=128
# num_label=30
# base_vec=~/work/data/filterbenchmark/LAION1M/LAION1M_base.fvecs
# # base_labelset=./attrdata/label/LAION1M_base_n1000448_l${num_label}_zipf.txt
# base_labelset=~/work/data/filterbenchmark/LAION1M/label_base.txt
# space="l2"


# db="tripclick"
# m=16
# efc=128
# num_label=29
# base_vec=~/work/data/filterbenchmark/tripclick/tripclick_base.fvecs
# base_labelset=~/work/data/filterbenchmark/tripclick/label_base.txt
# # base_vec=~/work/data/filterbenchmark/tripclick/tripclick_base.fvecs
# # base_labelset=/root/work/lssg/example/attrdata/label/tripclick_base_n1055976_l29_uniform.txt
# space="l2"

# db="glove"
# m=16
# efc=200
# num_label=2000
# base_vec=~/work/data/vecs/glove-100/glove-100_base.fvecs
# base_labelset=./attrdata/label/glove_base_n1183514_l${num_label}_zipf.txt
# space="l2"


# db="ytb_video"
# m=16
# efc=128
# num_label=3862
# base_vec=~/work/data/filterbenchmark/ytb_video/ytb_video_base.fvecs
# base_labelset=~/work/data/filterbenchmark/ytb_video/label_base.txt
# space="l2"

# db="ytb_audio"
# m=16
# efc=128
# num_label=3862
# base_vec=~/work/data/filterbenchmark/ytb_audio/ytb_audio_base.fvecs
# base_labelset=~/work/data/filterbenchmark/ytb_audio/label_base.txt
# space="l2"


# db="yfcc"
# m=16
# efc=128
# num_label=65521
# base_vec=~/work/data/filterbenchmark/yfcc/yfcc_base.fvecs
# base_labelset=~/work/data/filterbenchmark/yfcc/label_base.txt
# space="l2"

# db="yfcc-10m"
# m=16
# efc=128
# num_label=200386
# base_vec=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/base.10M.fvecs
# base_labelset=/root/work/others/big-ann-benchmarks/data/yfcc10M_converted/label_base_filled.txt
# space="l2"

# db="wikipedia-5m"
# m=16
# efc=128
# num_label=237445
# base_vec=/root/work/data/vecs/wikipedia/wikipedia_5m_base.fvecs
# base_labelset=/root/work/data/vecs/wikipedia/wikipedia_5m_base_labels.txt
# space="l2"

index="./index/${db}_${m}_${efc}_${space}_label${num_label}.poi"
scope="./index/${db}_label${num_label}.sco"


threads=16
mkdir -p "$(dirname $index)"

cd ../build
make clean
cmake .. && make -j8
cd ../example

../build/bin/build_po_label \
    --m $m \
    --efc $efc \
    --basevec $base_vec \
    --labelset $base_labelset \
    --space $space \
    --index_location $index \
    --scope_location $scope \
    --threads $threads
