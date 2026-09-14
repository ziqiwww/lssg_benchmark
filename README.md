# [SIGMOD '27] Fast Label-Filtering Approximate Nearest Neighbor Search via Progressive Label Set Stratification [The benchmarking repository].

> > This is the benchmarking library of LSSG, for the header-only library with python bindings, please check [here](https://github.com/nju-websoft/LSSG.git).


LSSG is a multi-tier proximity graph for label-filtering approximate nearest neighbor search (LFANNS). It unifies the three canonical label filters — **equality**, **containment**, and **overlap** — in a single index by interpreting label set relations through label set distance (Jaccard distance). Edges are stratified into tiers with nested similarity bounds (`(0,1] ⊃ (0, 0.67] ⊃ (0, 0.33] ⊃ (0,0]`); the bottom tier is label-agnostic for global navigability, while upper tiers connect increasingly similar label sets and preserve equality exactly at the top tier. Search performs tier-ordered in-filtering beam search, escalating to stricter tiers to escape local minima in selective regions.

Key contributions:

- A unified multi-tier graph supporting equality, containment, and overlap filters through progressive label set stratification.
- Incremental insertion with dual-space pruning (RNG-style pruning in vector space + label diversification in label space), and two similar-label-set selection strategies: progressive **IVF scanning** (LSSG-IVF, budget φ) and **MinHash LSH banding** (LSSG-MinHash, signature length τ, bands β) for scalability to large label spaces.
- Stepwise analysis of filter-valid expansion probabilities: equality is preserved exactly at the top tier, while containment and overlap admit lower bounds that strengthen monotonically with label set similarity, plus a conditional logarithmic bound on expected query cost.
- On eight real-world datasets, LSSG achieves ideal optimality for equality queries, and at matched accuracy is **1.06x–92.9x (avg. 19.3x) faster for containment** and **1.08x–84.1x (avg. 15.5x) faster for overlap** than the strongest competing index (ELI), using **0.35x index size** and **0.87x indexing time**.

## Requirements
1. C++20 compiler and CMake >= 3.20
2. OpenMP
3. (Optional) AVX-512 for the SIMD-accelerated bitset paths

## Datasets

### Dataset file format

- **Vectors (`.fvecs`)**: each record consists of a 4-byte integer dimension `d`, followed by `d` floating-point values (`4*d` bytes in total for the vector payload).

- **Ground truth (`.ivecs`)**: uses the same record layout as `.fvecs`, but stores integer values instead of floating-point values. These integers indicate ground-truth vector IDs.

- **Labels (`.txt`)**: each line is one label set, represented as comma-separated integer labels. The label file contains one label set per base/query vector.

- **Label files used in experiments**: one base label file and three query label files for the **equality**, **containment**, and **overlap** scenarios (e.g., `*_equality.txt`, `*_containment.txt`, `*_overlap.txt`).

### Dataset sources and descriptions

| Dataset | Size | Dim | \|A\| | \|ℒ\| | Description |
|---|---|---|---|---|---|
| [SIFT](http://corpus-texmex.irisa.fr/) | 1,000,000 | 128 | 12 | 2,385 | Classic ANNS benchmark [3]; synthetic Zipf labels generated following [5, 40, 58, 66] |
| [GIST](http://corpus-texmex.irisa.fr/) | 1,000,000 | 960 | 12 | 2,385 | Classic ANNS benchmark [3]; synthetic Zipf labels |
| [TripClick](https://tripdatabase.github.io/tripclick/) | 1,055,976 | 768 | 29 | 7,734 | Click logs from a health web search engine [47, 48], embedded by DPR [26]; 28 clinical-area labels + "others" |
| [LAION](https://laion.ai/blog/laion-400-open-dataset/) | 1,000,448 | 512 | 30 | 62,066 | Image–caption pairs [50] embedded by CLIP [46]; 1M subset with 30 caption keywords as labels |
| [YTB-Video](https://research.google.com/youtube8m/index.html) | 1,000,000 | 1,024 | 3,862 | 160,018 | YouTube-8M [49] topic labels; 1M video feature vectors |
| [YTB-Audio](https://research.google.com/youtube8m/index.html) | 5,000,000 | 128 | 3,862 | 498,335 | YouTube-8M [49] topic labels; 5M audio samples |
| [Wikipedia](https://huggingface.co/datasets/maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2) | 5,000,000 | 384 | 237,417 | 184,298 | English paragraph embeddings [18] via all-MiniLM-L6-v2 [61]; labels from Wikidata P31 (instance of) and P279 (subclass of) |
| [YFCC-1M](https://big-ann-benchmarks.com/neurips23.html) | 1,000,000 | 192 | 181,931 | 636,356 | Big-ANN NeurIPS'23 [38, 53]; main experiments use the 1M subset, YFCC-10M (200,386 labels, ~4M label sets) for the scalability benchmark |

## Baselines

Evaluated using their official open-source implementations with recommended parameters:

- **LSSG-IVF** / **LSSG-MinHash**: our two variants, differing only in similar label set selection. Unless stated otherwise, both use T=9 tiers, m=16, ω_c=128, with φ=50,000 for IVF and (τ, β)=(64, 16) for MinHash.

- **ACORN** ([https://github.com/guestrin-lab/ACORN](https://github.com/guestrin-lab/ACORN)): predicate-agnostic index; M=16, M_β=32, L=256, γ=30.

- **Packing** ([https://github.com/SpaceIshtar/FilterGraph](https://github.com/SpaceIshtar/FilterGraph)): the optimal strategy to solve LFANNS; M=96, efCons=256.

- **RWalks** ([https://github.com/anon-sigmod/RWalks/tree/main](https://github.com/anon-sigmod/RWalks/tree/main)): label vector + hybrid distance; M=32, efConstruction=200, τ=0, h=0.1.

- **UNG** ([https://github.com/YZ-Cai/Unified-Navigating-Graph](https://github.com/YZ-Cai/Unified-Navigating-Graph)): separate Vamana indices per label set + cross-group edges; M=32, L=100, δ=6 cross-group edges.

- **ELI** ([https://github.com/mingyu-hkustgz/LabelANN](https://github.com/mingyu-hkustgz/LabelANN)): frequent-itemset-based index sharing; M=16, efConstruction=200, elastic factor 0.2.

- **ANNS libraries & VDBMS**: [FAISS](https://github.com/facebookresearch/faiss) and [VSAG](https://github.com/antgroup/vsag) as ANNS libraries; [Milvus](https://github.com/milvus-io/milvus) (representative specialized VDBMS) and [PGVector](https://github.com/pgvector/pgvector) (representative relational DBMS with ANNS support) as vector databases with native filtering; HNSW (M=32, efCons=200) and k-means IVF (n_lists=10,000); **pre-filtering** (exact linear scan of the filtered subset) and the **oracle label-filtered RNG** (HNSW on the fully filtered subset) as references.

## Evaluation

### Setup

Experiments are provided as bash scripts under `./example`:

- `build_polabel.sh`: builds LSSG on a dataset. The IVF and MinHash variants are switched by the conditional compile option — comment or uncomment `#define USE_MINHASH` in `lssg/scope_label.hh`.
- `search_polabel.sh`: searches LSSG-IVF / LSSG-MinHash indices built and saved by the build script.

All figures in the paper can be generated with the python scripts in `example/plot/`.

### Summary of experimental results (Section 6 of the paper)

- **RQ1 – Indexing (Table 4, 16-thread parallelism)**: LSSG-IVF / LSSG-MinHash use 0.82x / 0.84x of UNG's index size (the most space-efficient baseline), and LSSG-MinHash achieves the best or near-best indexing time on all datasets (0.78x UNG, 0.87x ELI). Indexing cost stays stable as the label space grows, unlike label-sharing baselines (ELI, Packing, RWalks).
- **RQ2 – Query performance (Figs. 4–7)**: at matched recall@10, LSSG is 1.06x–92.9x (avg. 19.3x) faster for containment and 1.08x–84.1x (avg. 15.5x) faster for overlap than the strongest competitor ELI, while being the most stable method across selectivity percentiles (10%–90%) and approximating the oracle label-filtered RNG with the fewest distance computations. For equality, LSSG matches the best baseline (UNG) while supporting native incremental updates.
- **Robustness (Figs. 8–10)**: LSSG stays superior when the label space |A| grows (up to 8,192 labels, where ELI and UNG fail), across Zipf / Uniform / Poisson / Multinomial label distributions, and under insertion drift (unseen labels and frequency inversion after indexing the first 50%), with insertion throughput of 3,010–5,668 vectors/s.
- **Micro benchmarks (Figs. 11–16, Table 5)**: tier usage concentrates on upper tiers for selective filters; the filter-valid expansion ratio rises from ~0.1 to 1.0 with tier strictness; LabelPrune reduces indexing time by up to 51% at <10% query cost; a moderate IVF budget φ=50,000 is near-optimal; on YFCC-10M, LSSG-MinHash reduces indexing time by 62% vs. LSSG-IVF and scales as O(|V|) index size / O(|V| log |V|) indexing time.
