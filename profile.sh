#! /bin/sh
if [ -z "$3" ]; then
    filename=$(/usr/local/cuda/bin/ncu -f --set full -o $1-%i $2 | tee /dev/fd/2 | grep "Report:" | awk '{print $3}')
else
    filename=$(/usr/local/cuda/bin/ncu -f --set full -o $1-%i $2 $3 $4 $5 $6| tee /dev/fd/2 | grep "Report:" | awk '{print $3}')
fi

/usr/local/cuda/bin/ncu-ui --shared-instance 1 ${filename}

