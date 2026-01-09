import numpy as np
import torch
import random
import os
import math

from transformers.data.data_collator import pad_without_fast_tokenizer_warning


class InferenceCollator:
    def __init__(self, tokenizer):
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.padding = "longest"

    def __call__(self, features):
        feature_names = ["input_ids", "idx", "attention_mask"]

        no_labels_features = [
            {k: v for k, v in feature.items() if k in feature_names} for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["answer"] = [features[i]["answer"] for i in range(len(features))]
        return batch


def add_noise(
    x: torch.Tensor,
    std: float = 0.1,
    mask: torch.Tensor | None = None,
):
    """
    Add noise to vector x.

    Args:
    x (torch.Tensor): Input vector (can be of any shape).
    std (float): Standard deviation or amplitude of the noise.
    mask (torch.Tensor): (optional) Apply noise only to elements where mask==1.

    Returns:
    torch.Tensor: Noisy vector
    """
    noise = torch.randn_like(x) * std

    if mask is not None:
        noise = noise * mask.to(noise.dtype).to(x.device)
    x_noisy = x + noise

    return x_noisy


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            count += 1
            m.train()
    if count == 0:
        raise ValueError("No dropout layers found in the model")


def disable_dropout(model):
    """Function to disable the dropout layers during test-time"""
    count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            count += 1
            m.eval()

    if count == 0:
        raise ValueError("No dropout layers found in the model")


def set_dropout_p(model, p: float, verbose: bool = False):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
            if verbose:
                print(f"Set {module} p from {module.p} to {p}")
            module.p = p


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pass_at_k_mean(corrects, k):
    """
    This function calculates the mean pass@k under |D| inputs, each with N sampled answers.

    Args:
        corrects: numpy array of shape (|D|, N), binary (1 if correct, 0 if not)
        k: int

    Returns:
        pass_at_k: float
    """
    num_samples = corrects.shape[1]
    if isinstance(corrects, torch.Tensor):
        corrects = corrects.cpu().numpy().astype(np.int32)
    c = np.sum(corrects, axis=1)  # number of correct samples per input

    # Calculate the binomial coefficients safely
    def comb(n, r):
        if r > n or r < 0:
            return 0
        return math.comb(n, r)

    # Calculate pass@k for each input
    pass_at_k_scores = []
    for i in range(corrects.shape[0]):
        c_i = c[i]  # number of correct samples for input i

        # If k > num_samples, we can't choose k samples, so pass@k = 1 if any correct exists
        if k > num_samples:
            pass_at_k_i = 1.0 if c_i > 0 else 0.0
        # If no correct samples, pass@k = 0
        elif c_i == 0:
            pass_at_k_i = 0.0
        else:
            # Probability of choosing k samples with zero correct answers:
            fail_prob = comb(num_samples - c_i, k) / comb(num_samples, k)
            pass_at_k_i = 1 - fail_prob

        pass_at_k_scores.append(pass_at_k_i)

    # Return the mean pass@k across all inputs
    return np.mean(pass_at_k_scores)
