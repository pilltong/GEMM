#!/usr/bin/env bash

set -u

# Define the range of values for each parameter
BK_VALUES=(8 16 32 64)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
NUM_THREADS_VALUES=(256 512 1024)

cd ".."

RUNNER="runner.cuh"
KERNEL="../headers/cuda_kernels/6_vectorized.cuh"
OUTPUT="benchmark_results/6_vectorized_t2.txt"

# Clear the output file
echo "" > $OUTPUT

# Set GPU to use
export DEVICE="0"

TOTAL_CONFIGS="$(( ${#NUM_THREADS_VALUES[@]} * ${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} ))"
CONFIG_NUM=0

# Loop through all combinations of parameters
for bk in ${BK_VALUES[@]}; do
  for tm in ${TM_VALUES[@]}; do
    for tn in ${TN_VALUES[@]}; do
      for bm in ${BM_VALUES[@]}; do
        for bn in ${BN_VALUES[@]}; do
          for nt in ${NUM_THREADS_VALUES[@]}; do
            echo ""
            CONFIG_NUM=$(( $CONFIG_NUM + 1 ))

            # skip configurations that don't fullfil preconditions
            config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NT=$nt"
            if [[ $(( ($nt * 4) % bk )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BK = $(( ($nt * 4) % bk )) != 0))"
              continue
            fi
            if [[ $(( ($nt * 4) % bn )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BN = $(( ($nt * 4) % bn )) != 0))"
              continue
            fi
            if [[ $(( $bn % (16 * $tn ) )) -ne 0 ]]; then
              echo "QUANTIZATION: Skipping $config because BN % (16 * TN) = $(( $bn % (16 * $tn ) )) != 0))"
              continue
            fi
            if [[ $(( $bm % (16 * $tm ) )) -ne 0 ]]; then
              echo "QUANTIZATION: Skipping $config because BM % (16 * TM) = $(( $bm % (16 * $tm ) )) != 0))"
              continue
            fi
            if [[ $(( ($bm * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (BM * BK) % (4 * NUM_THREADS) = $(( ($bm * $bk) % ( 4 * 256 ) )) != 0))"
              continue
            fi
            if [[ $(( ($bn * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (BN * BK) % (4 * NUM_THREADS) = $(( ($bn * $bk) % ( 4 * 256 ) )) != 0))"
              continue
            fi

            # Update the parameters in the source code
            sed -i "s/const uint k6_bk = .*/const uint k6_bk = $bk;/" $RUNNER
            sed -i "s/const uint k6_tw_m = .*/const uint k6_tw_m = $tm;/" $RUNNER
            sed -i "s/const uint k6_tw_n = .*/const uint k6_tw_n = $tn;/" $RUNNER
            sed -i "s/const uint k6_bm = .*/const uint k6_bm = $bm;/" $RUNNER
            sed -i "s/const uint k6_bn = .*/const uint k6_bn = $bn;/" $RUNNER
            #sed -i "s/const int K9_NUM_THREADS = .*/const int K9_NUM_THREADS = $nt;/" $KERNEL
            
            # Rebuild the program
            make 

            echo "($CONFIG_NUM/$TOTAL_CONFIGS)" |& tee -a $OUTPUT
            # Run the benchmark and get the result
            ./main 6 | tee -a $OUTPUT

            echo "BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NUM_THREADS=$nt" |& tee -a $OUTPUT
          done
        done
      done
    done
  done
done