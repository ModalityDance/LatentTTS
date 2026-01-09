import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.func = self._filter_ignore_index

    def _filter_ignore_index(self, input: torch.Tensor, target: torch.Tensor):
        """
        Filter out the logits and labels that are ignored.
        Args:
            input (torch.Tensor): The input logits, (B, S, D)
            target (torch.Tensor): The target labels, (B, S)
        Returns:
            tuple: A tuple of filtered logits and labels, (N, D), (N,)
            where N is the number of valid tokens of all sequences.
        """
        # logits (B, S, D) labels (B, S) mask (B, S)
        mask = target != self.ignore_index
        # mask (B, S, D)
        mask_expand = mask.unsqueeze(-1).expand_as(input)
        # logits (B, S, D) -> (N, D)
        logits_filtered = input[mask_expand]
        logits_filtered = logits_filtered.view(-1, input.shape[-1])

        # labels (B, S) -> (N,)
        labels_filtered = target[mask]
        labels_filtered = labels_filtered.view(-1).to(logits_filtered.dtype)

        return logits_filtered, labels_filtered

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # input: logits (B, S, 1), target: labels (B, S)
        logits_filtered, labels_filtered = self.func(input, target)
        loss = super().forward(logits_filtered.squeeze(-1), labels_filtered)

        return loss


class MaskedCrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, n_samples: int):
        # logits (B, seq_len, 1) -> (a_B, n_samples, seq_len), labels (B, seq_len) -> (a_B, n_samples, seq_len)
        actual_batch_size = input.shape[0] // n_samples
        logits_filtered = input.squeeze(-1).view(actual_batch_size, n_samples, -1)
        labels_filtered = target.view(actual_batch_size, n_samples, -1)
        # set the label on -100 postion
        _mask = labels_filtered == -100
        # labels_filtered = torch.where(_mask, 0.0, labels_filtered)
        # logits_filtered = torch.where(_mask, 0.0, logits_filtered)
        log_probs = F.log_softmax(logits_filtered, dim=1)
        # mask out the padding tokens
        log_probs = log_probs * ~_mask

        # labels_filtered = labels_filtered.softmax(dim=1)
        labels_filtered = labels_filtered * ~_mask
        # dim 1 (-2) is the n_samples dimension, which is also the "classification" dimension
        loss = -(labels_filtered * log_probs).sum(dim=1).mean()

        return loss
