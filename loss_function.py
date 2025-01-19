#loss_function.py

import torch
import torch.nn.functional as F

def train_loss(scores, labels, batch_size, lamda_sparsity=8e-5, lamda_smooth=8e-5, k_ratio=0.2):
    """
    Ranking loss with Binary Cross-Entropy for weakly-supervised MIL anomaly detection.
    Handles label-output consistency with top-k selection.
    """
    num_segments = scores.shape[0] // batch_size  # Segments per video
    total_loss = torch.tensor(0.0, requires_grad=True, device=scores.device)

    for i in range(batch_size):
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i].float()  # Single video-level label

        # Compute feature magnitudes
        feature_magnitudes = torch.norm(video_scores, p=2, dim=-1)

        # Top-k selection
        k = max(1, int(len(feature_magnitudes) * k_ratio))  # Select top k% of snippets
        top_k_features, _ = torch.topk(feature_magnitudes, k)

        # Compute BCE loss for the mean of top-k features
        mean_score = top_k_features.mean()
        bce_loss = F.binary_cross_entropy_with_logits(mean_score, video_label)

        # Sparsity and Smoothness
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))
        smoothness_loss = lamda_smooth * torch.sum(
            (torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2
        )

        # Combine all components
        total_loss = total_loss + bce_loss + sparsity_loss + smoothness_loss

    return total_loss / batch_size

def val_test_loss(scores, labels, batch_size, lamda_sparsity=8e-5, lamda_smooth=8e-5):
    """
    Ranking loss with Binary Cross-Entropy for weakly-supervised MIL anomaly detection.
    Uses video-level Binary Cross-Entropy instead of top-k selection.
    """
    num_segments = scores.shape[0] // batch_size  # Segments per video
    total_loss = torch.tensor(0.0, requires_grad=True, device=scores.device)

    for i in range(batch_size):
        # Extract the scores for this video
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i].float()  # Single video-level label

        # Compute the mean score for the video
        mean_score = video_scores.mean()

        # Binary Cross-Entropy loss for the video
        bce_loss = F.binary_cross_entropy_with_logits(mean_score, video_label)

        # Sparsity and Smoothness
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))
        smoothness_loss = lamda_smooth * torch.sum(
            (torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2
        )

        # Combine all components
        total_loss = total_loss + bce_loss + sparsity_loss + smoothness_loss

    return total_loss / batch_size
