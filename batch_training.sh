#!/bin/bash

for depth in 2 4 6 8
do
    for width_factor in 16 32 64
    do
        sbatch training.slurm $depth $width_factor;
        echo "Submitted job for depth = $depth, width_factor = $width_factor"
    done
done