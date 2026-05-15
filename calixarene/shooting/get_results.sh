#!/bin/bash

for g in G2 G5 G6
do
    echo $g
    for i in $(seq 0 5)
    do
        echo -e "$i: \c"
        head -qn 2 "$g"_1/$i/*/STOPCAR 2> /dev/null | \
            grep -v '#' | \
            awk 'BEGIN{i=0;c=0;}{if($NF>0){i+=($3-1);c+=1};}END{printf("%8.5f %3d %3d\n", i/c, i, c - i);}'
    done

    for i in $(seq 6 11)
    do
        echo -e "$i: \c"
        head -qn 2 "$g"_2/$i/*/STOPCAR 2> /dev/null | \
            grep -v '#' | \
            awk 'BEGIN{i=0;c=0;}{if($NF>0){i+=($3-1);c+=1};}END{printf("%8.5f %3d %3d\n", i/c, i, c - i);}'
    done
done
