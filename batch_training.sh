#!/bin/bash

for depth in 2 3 4
do
    for width_factor in 96 128
    do
        sbatch training.slurm $depth $width_factor;
        echo "Submitted job for depth = $depth, width_factor = $width_factor"
    done
done