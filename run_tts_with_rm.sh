#!/usr/bin/env zsh

# Enable zsh options for better error handling
set -e
set -u

# Variables
seed=0
script=src.infer_gpt2_rm
prm_id="checkpoints/latentRM"


splits=(gsm_test gsm_hard multiarith)
num_beams_array=(1 2 4 8)
beam_candidates=(16 8 4 2)
for split in $splits; do
    echo "######### $split #########"
    for num_beams in $num_beams_array; do
        log_file="BS_${split}_beams=${num_beams}.log"
        basic_args="--seed=$seed prm_mode=beam_search --num_beams=$num_beams --data_path=data/${split}.json --prm_id=$prm_id"
        for candidates in $beam_candidates; do
            echo "--$basic_args --num_beam_candidates=$candidates" >> $log_file
            python -m $script $basic_args --num_beam_candidates=$candidates >> $log_file
            echo "--------------------------------" >> $log_file
        done
    done
done

n_return_sequences_array=(1 4 16 64)
for n_return_sequences in $n_return_sequences_array; do
    log_file="BoN_${split}_n=${n_return_sequences}.log"
    basic_args="--seed=$seed prm_mode=best_of_n --num_return_sequences=$n_return_sequences --data_path=data/${split}.json --prm_id=$prm_id"
    python -m $script $basic_args >> $log_file
    echo "--------------------------------" >> $log_file
done