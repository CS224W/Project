#!/bin/bash

cd ~/Documents/Pers/MMDS/Course/CS224W/Project/snap-master/examples/node2vec/

options=( 0.25 0.50 1 2 4 )

input_fl="/Users/prmathur/Documents/Pers/MMDS/Course/CS224W/Project/code/code/data_created/graphs/user-user.txt"
input_fl_ut="/Users/prmathur/Documents/Pers/MMDS/Course/CS224W/Project/code/code/data_created/graphs/user-tag.txt"
output_fl_loc="/Users/prmathur/Documents/Pers/MMDS/Course/CS224W/Project/snap-master/examples/node2vec/emb"

for p in ${options[@]}
do
    for q in ${options[@]}
    do
        echo "p $p, q $q"
        # echo "$output_fl_loc/uu_p$p-q$q.emb"
        # ./node2vec -i:$input_fl -o:"$output_fl_loc/uu_p$p-q$q.emb" -p:$p -q:$q -v -e:5
        ./node2vec -i:$input_fl_ut -o:"$output_fl_loc/ut_p$p-q$q.emb" -p:$p -q:$q -v -e:5
    done
done
