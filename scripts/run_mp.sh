#!/bin/bash
echo `which python`
cd ..

acquisition="max-optimism"
experiment="DeepAL-8GPUS-Sample-Run"
# experiment="DeepAL-8GPUS-Sample-Run-fixedembedding"
# experiment="DeepAL-8GPUS-Sample-Run-noinitialtraining"
# experiment="DeepAL-8GPUS-Sample-Run-freeembedding"

config="configs/active_representation_learning/active_represent_mp.json"
# config="configs/active_representation_learning/active_reprsent_mp_fixedembedding.json"
# config="configs/active_representation_learning/active_represent_mp_noinitialtraining.json"
# config="configs/active_representation_learning/active_represent_mp_freeembedding.json"

n_samples=100
n_rounds=5


firsthost=$(jsrun --nrs 1 -r 1 hostname)
export MASTER_ADDR=$firsthost
export MASTER_PORT=$((12345))

for replicate in `seq 0 0`
do
    infile="data/interaction_vector_r$replicate.npy"
    outfile="outputs/"$experiment"_batchsize_"$n_samples"_"numrounds_$n_rounds"_r"$replicate".csv"
    echo "Replicate $replicate"
    echo "Acquisition $acquisition"
    echo "outfile $outfile"

    # this set up is for 2 nodes where each node has 4 GPUS, modify this line accordingly to your 
    # computing system's setup
    lrun -n 8 -N 2 python run_active_learning_mp.py --interaction_file $infile --model active_representation --config_file $config --outfile $outfile  --n_samples $n_samples --n_rounds $n_rounds --acquisition $acquisition --seed $replicate
done
wait