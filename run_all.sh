#!/bin/bash

# Define the list of fitness data, exploration algorithms, and seeds to loop over
# EXPLORATION_ALGORITHMS=("batchbo" "random" "pex" "antbobatch")
EXPLORATION_ALGORITHMS=("batchbo" "random" "pex")

SEEDS=(1 2 3)

# Loop over the different combinations of parameters and run the command in parallel

for ea in "${EXPLORATION_ALGORITHMS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        # Run the command with the current set of parameters in the background
        python run.py \
            --device 'cpu' \
            --oracle_model antibody \
            --alg "$ea" \
            --name "${ea}-bsa-$seed" \
            --num_rounds 10 \
            --net cnn \
            --ensemble_size 3 \
            --out_dir resultantibody \
            --seed "$seed" \
            & # Run command in background
    done
done

wait # Wait for all background processes to finish before continuing
# pkill -f run.py
# chmod +x