#!/bin/bash

for depth in 6 8 10
do
    for width_factor in 128 256 512 1024
    do
        sbatch training.slurm $depth $width_factor;
        echo "Submitted job for depth = $depth, width_factor = $width_factor"
    done
done