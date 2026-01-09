#!/bin/bash
CUDA_VISIBLE_DEVICES=0

default_params=(
    "--save_path=latent-data/coconut"
    "--batch_size=32"
    "--model_type=coconut "
    "--seed=42"
    "--generation_latent_do_sample_by=dropout"
    "--generation_dropout_p=0.2"
    "--use_wandb=False"
)
mkdir -p latent-data/coconut
# training data
accelerate launch --main_process_port 0 --num_processes 1 -m src.annotate_data \
    ${default_params[@]} \
    --n_samples_per_step=64 \
    --n_samples=8 \
    --data_path=data/gsm_train.json \
    --name="train"


# validation data for evaluation
Ns=(4 64)
for N in ${Ns[@]}; do
    accelerate launch --main_process_port 0 --num_processes 1 -m src.annotate_data \
        ${default_params[@]} \
        --n_samples_per_step=0 \
        --n_samples=${N} \
        --data_path=data/gsm_valid.json \
        --name="valid-${N}"
done