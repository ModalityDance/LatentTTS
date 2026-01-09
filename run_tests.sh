#!/usr/bin/zsh


# Enable zsh options for better array handling
setopt extended_glob
setopt null_glob

grep_template="Pass@|Coverage:|Voting Accuracy:"

# Define arrays
typeset -a SEEDs=(0)
typeset -a Ns=(64)
typeset -a batch_sizes=(8)
typeset -a SPLITS=(gsm_test gsm_hard multiarith)

export CUDA_VISIBLE_DEVICES=6
# typeset -a DROPOUT_PS=(0.2)
# typeset -a NOISE_STDS=(0.6)
# MODEL_TYPE="coconut"

# export CUDA_VISIBLE_DEVICES=5
# typeset -a DROPOUT_PS=(0.2)
typeset -a DROPOUT_PS=()
typeset -a NOISE_STDS=(1.4)
MODEL_TYPE="codi"



 echo "MODEL_TYPE=$MODEL_TYPE"
# Create results directory if it doesn't exist
[[ ! -d results ]] && mkdir -p results

# Function to run inference and capture results
run_inference() {
    local split=$1
    local sampling_method=$2
    local param_name=$3
    local param_value=$4
    local txt_file="results/${MODEL_TYPE}_${split}_${param_name}_${param_value}.txt"
    
    for seed in $SEEDs; do
        echo "-------------Next Trial-------------------" >> $txt_file
        echo "-------------Next Trial-------------------"
        
        for i in {1..${#Ns}}; do
            local batch_size=$batch_sizes[$i]
            local N=$Ns[$i]
            echo "batch_size=$batch_size, N=$N, seed=$seed, sampling_method=$sampling_method, param_name=$param_name, param_value=$param_value"
            
            # Create temporary file with process ID
            local temp_file="/tmp/output_$$"
            
            # Run inference based on sampling method
            if [[ $sampling_method == "dropout" ]]; then
                python -m src.infer_gpt2 \
                    --data_path="data/${split}.json" \
                    --model_type="$MODEL_TYPE" \
                    --do_sample=True \
                    --sampling_by=dropout \
                    --batch_size=$batch_size \
                    --dropout_p=$param_value \
                    --n_samples=$N 2>&1 | tee $temp_file
            else
                python -m src.infer_gpt2 \
                    --data_path="data/${split}.json" \
                    --model_type="$MODEL_TYPE" \
                    --do_sample=True \
                    --sampling_by="noise" \
                    --batch_size=$batch_size \
                    --noise_std=$param_value \
                    --n_samples=$N 2>&1 | tee $temp_file
            fi
            
            # Extract results and clean up
            local results=$(grep -E "$grep_template" $temp_file)
            rm -f $temp_file
            echo "N=$N $results" >> $txt_file
            echo ""
            echo ""
        done
    done
}

# Run dropout experiments
for split in $SPLITS; do
    for dropout_p in $DROPOUT_PS; do
        run_inference $split "dropout" "dropout" $dropout_p
    done
done

# Run noise experiments
for split in $SPLITS; do
    for noise_std in $NOISE_STDS; do
        run_inference $split "noise" "noise" $noise_std
    done
done
