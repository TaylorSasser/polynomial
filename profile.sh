#! /bin/sh
if [ -z "$3" ]; then
    filename=$(ncu -f --set full -o $1-%i $2 | tee /dev/fd/2 | grep "Report:" | awk '{print $3}')
else
    filename=$(ncu -f --set full -o $1-%i $2 $3 | tee /dev/fd/2 | grep "Report:" | awk '{print $3}')
fi

ncu-ui --shared-instance 1 ${filename}

