#!/bin/sh

# Save the first two arguments separately
output_prefix="$1"
binary="$2"

# Use 'shift' twice to remove the first two arguments from the list
shift 2

# Now "$@" will contain all remaining arguments
# Pass these to the ncu command, excluding the first two script arguments
filename=$(/usr/local/cuda/bin/ncu -f --set full -o "${output_prefix}" "$binary" "$@" | tee /dev/fd/2 | grep "Report:" | awk '{print $3}')

