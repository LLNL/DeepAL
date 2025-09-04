#!/bin/bash
echo `which python`
cd ..

experiment="DeepAL-Sample-Run"
acquisition="eps-greedy"

config="configs/active_representation_learning/active_represent.json"
n_samples=100
n_rounds=5


for replicate in `seq 0 9`
do
    infile="data/interaction_vector_r$replicate.npy"
    outfile="outputs/Sample-Run_batchsize_"$n_samples"_"numrounds_$n_rounds"_r"$replicate".csv"


    echo "Acquisition $acquisition"
    echo "Replicate $replicate"
    echo "outfile $outfile"
    python run_active_learning.py --interaction_file $infile --model active_representation --config_file $config --outfile $outfile  --n_samples $n_samples --n_rounds $n_rounds --acquisition $acquisition --seed $replicate --gpu 0
done
wait