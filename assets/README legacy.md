# latentTTS

## Overview

This repository provides implementation code for the paper [**Parallel Test-Time Scaling for Latent Reasoning Models**](https://arxiv.org/abs/2510.07745v1). It includes two stochastic sampling methods for continuous thoughts, Monte Carlo Dropout and Additive Gaussian Noise, and a LatentRM for best-of-$N$ and beam search.  

**Backbones covered** COCONUT, CODI, CoLaR. **Benchmarks** GSM8K Test, GSM8K Hard, MultiArith. 


## Project Structure

```
latent-tts/
├── src/                   # Source code
│   ├── models/            # Model implementations
│   ├── annotate_data.py   # Data annotation script
│   ├── train.py           # latentRM training script
│   ├── infer_gpt2.py      # GPT-2 inference
│   ├── infer_llama.py     # LLaMA inference
│   └── infer_gpt2_rm.py   # latentRM-based inference
├── training_args/         # Training configurations
├── data/                  # Dataset files
├── checkpoints/           # Model checkpoints
│   └── latentRM/          # latentRM checkpoint (👉 checkpoint for COCONUT is available at https://huggingface.co/dd101bb/latentRM)
├── utils/                 # Utility functions
└── run_*.sh               # Execution scripts
```

## Quick Start

### 1. Data Annotation

First, run the data annotation process to prepare training data:

```bash
./run_annotation.sh
```

This script will:
- Process training data and validation data with specified batch size and sampling parameters
- Generate annotated data for LatentRM training
- Save results to the specified output directory

### 2. Training Configuration

Configure your training parameters in the `training_args/` directory. The main configuration file is `train_coconut.yaml`:

```yaml
run_name: "run1"
metric_for_best_model: "test_n_64_recall_at_1"
output_dir: "/workspace/model-out/"
# ... other parameters
```

### 3. Model Training

Navigate to your project directory and launch training:

```bash
cd your/path/to/latent-tts
accelerate launch -m src.train training_args/train_coconut.yaml
```

The training process will:
- Load the annotated data from the previous step
- Train the latentRM with the specified configuration
- Save checkpoints and evaluation results

### 4. Evaluation and Testing

#### Majority Voting and Coverage Testing

Run comprehensive evaluation using majority voting and coverage metrics:

```bash
# For LLaMA model (CoLaR)
./run_tests_llama.sh

# For GPT-2 models (COCONUT and CODI)
./run_tests.sh
```

These scripts will:
- Test different sampling strategies (dropout, noise)
- Evaluate on multiple datasets (GSM-Test, MultiArith, GSM-Hard)
- Generate detailed performance metrics including Pass@k, Coverage, and Voting Accuracy

####  Testing

For beam search evaluation:

```bash
./run_tts_with_rm.sh
```

This script will:
- Test beam search with different `beam size` (1, 2, 4, 8)
- Test Best-of-N with different `n_return_sequences` (1, 4, 16, 64)
- Generate logs for different configurations

## Hugging Face Paper Page


📄 Explore the paper on [**Hugging Face Papers**](https://huggingface.co/papers/2510.07745) —  
it includes community discussions, citation tools, and related resources.

⭐ If you find our work insightful, please consider giving it an **upvote** to support further research!

## Citation

If you find this repository useful, please cite the paper.

```
@misc{you2025paralleltesttimescalinglatent,
      title={Parallel Test-Time Scaling for Latent Reasoning Models}, 
      author={Runyang You and Yongqi Li and Meng Liu and Wenjie Wang and Liqiang Nie and Wenjie Li},
      year={2025},
      eprint={2510.07745},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.07745}, 
}
```
