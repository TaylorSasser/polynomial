#!/bin/bash

# Path to the CSV file
output_csv="./results/metrics.csv"

rm $output_csv
touch $output_csv

# Selecting 5 representative degrees between 1 and 50000
declare -a degrees=("1" "12500" "25000" "37500" "50000")

# Bytes in a gigabyte
bytes_in_gb=$((2**30))


# Selecting 5 representative sizes between 1 GB and 12 GB (in terms of number of floats)
declare -a sizes=("1", "1000" "100000" "1000000" "1000000000", "3000000000")

# Metrics array
declare -a metrics=("gpu__time_duration.sum", "sm__cycles_elapsed.avg.per_second", "dram__bytes.sum.per_second", "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained")

metrics_csv=$(IFS=,; echo "${metrics[*]}")

echo '' > $output_csv

# Loop over degrees and sizes
for degree in "${degrees[@]}"; do
    for size in "${sizes[@]}"; do
        # Run the polynomial calculation and append output to the CSV file
        /usr/local/cuda/bin/ncu --csv --metrics $metrics_csv ./build/release/polynomial $degree $size >> $output_csv
    done
done
