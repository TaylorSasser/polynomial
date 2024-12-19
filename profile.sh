#!/bin/sh

output_prefix="$1"
binary="$2"

shift 2

filename=$(/usr/local/cuda/bin/ncu -f --set full -o "${output_prefix}" "$binary" "$@" | tee /dev/fd/2 | grep "Report:" | awk '{print $3}')

