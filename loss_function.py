#loss_function.py

import torch
import torch.nn.functional as F

def ranking_loss(scores, labels, batch_size, lamda_sparsity=8e-5, lamda_smooth=8e-5, margin=1.0):
    """
    Updated Ranking loss for weakly-supervised MIL anomaly detection.
    """
    num_segments = scores.shape[0] // batch_size  # Segments per video
    total_loss = torch.tensor(0.0, requires_grad=True, device=scores.device)

    for i in range(batch_size):
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i].item()  # Extract the scalar value

        # Compute max scores
        if video_scores.numel() == 0:  # Handle empty video_scores
            max_anomalous = torch.tensor(float('-inf'), device=scores.device)
            max_normal = torch.tensor(float('-inf'), device=scores.device)
        elif video_label == 0:  # Anomalous video
            max_anomalous = video_scores.max()
            max_normal = torch.tensor(float("-inf"), device=scores.device)
        elif video_label == 1:  # Normal video
            max_anomalous = torch.tensor(float("-inf"), device=scores.device)
            max_normal = video_scores.max()
        else:
            raise ValueError(f"Invalid video_label: {video_label}")

        # Ranking loss
        if max_anomalous != float("-inf") and max_normal != float("-inf"):
            rank_loss = F.relu(margin - max_anomalous + max_normal)
        else:
            rank_loss = torch.tensor(0.0, device=scores.device)

        # Sparsity and Smoothness losses
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))
        smoothness_loss = lamda_smooth * torch.sum(
            (torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2
        )

        total_loss = total_loss + rank_loss + sparsity_loss + smoothness_loss

    return total_loss / batch_size

if __name__ == "__main__":
    scores = torch.rand(16, requires_grad=True)
    labels = torch.tensor([0, 1, 1, 0])
    loss = ranking_loss(scores, labels, batch_size=4)
    print(f"Loss: {loss}, Requires Grad: {loss.requires_grad}")
