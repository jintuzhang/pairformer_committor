#!/bin/bash

for i in $(seq 0 17)
do
    echo -e "$i: \c"
    head -qn 2 ../shooting/results/$i/*/STOPCAR 2> /dev/null | \
        grep -v '#' | \
        awk 'BEGIN{i=0;c=0;}{if($NF>0){i+=($4-1);c+=1};}END{if(c>0){printf("%8.5f %3d %3d\n", i/c, i, c - i)}else{print("")};}'
done
