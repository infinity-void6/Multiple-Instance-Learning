import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from split_data_train_val_test import create_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from Sequential_Model import AnomalyAutoencoder
from loss_function import ranking_loss

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model
model = AnomalyAutoencoder(input_dim=2048).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
criterion = ranking_loss
scaler = GradScaler()

# Data Loaders
train_loader = create_dataloader("split/train/normal", "split/train/anomalous", batch_size=4, max_segments=1018)
val_loader = create_dataloader("split/val/normal", "split/val/anomalous", batch_size=4, max_segments=282)
test_loader = create_dataloader("split/test/normal", "split/test/anomalous", batch_size=4, max_segments=304)


def process_in_chunks(inputs, model, chunk_size, device):
    """
    Process inputs in smaller chunks to reduce memory usage.
    """
    outputs = []
    for i in range(0, inputs.size(0), chunk_size):
        chunk = inputs[i:i+chunk_size].to(device)
        with torch.no_grad():
            output = model(chunk)
        outputs.append(output)
    return torch.cat(outputs, dim=0)


def train_epoch(train_loader, model, optimizer, criterion, device, scaler, chunk_size):
    model.train()  # Set model to training mode
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for normal_features, anomalous_features in tqdm(train_loader, desc="Training"):
        # Combine features
        if normal_features.dim() > 2:
            normal_features = normal_features.squeeze(-1).squeeze(-1).squeeze(-1)
            normal_features = normal_features.view(-1, normal_features.size(-1))
        if anomalous_features.dim() > 2:
            anomalous_features = anomalous_features.squeeze(-1).squeeze(-1).squeeze(-1)
            anomalous_features = anomalous_features.view(-1, anomalous_features.size(-1))

        normal_features = normal_features.to(device)
        anomalous_features = anomalous_features.to(device)
        inputs = torch.cat((normal_features, anomalous_features), dim=0)

        # Labels: 1 for normal, 0 for anomalous
        labels = torch.cat((
            torch.ones(len(normal_features), 1).to(device),
            torch.zeros(len(anomalous_features), 1).to(device)
        ), dim=0)

        optimizer.zero_grad()

        # Process in chunks to handle memory constraints
        num_chunks = (inputs.size(0) + chunk_size - 1) // chunk_size  # Calculate number of chunks
        chunk_losses = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, inputs.size(0))
            input_chunk = inputs[chunk_start:chunk_end]
            label_chunk = labels[chunk_start:chunk_end]

            # Mixed precision forward pass for the chunk
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True):
                outputs_chunk = model(input_chunk)
                loss_chunk = criterion(outputs_chunk, label_chunk, len(label_chunk))  # Use len(label_chunk) as batch_size

            if torch.isnan(loss_chunk) or loss_chunk == 0.0:
                print(f"Skipping chunk {chunk_idx} due to invalid loss: {loss_chunk}")
                continue

            scaler.scale(loss_chunk).backward()  # Backpropagate chunk loss
            chunk_losses.append(loss_chunk.item())

            all_probs.extend(outputs_chunk.squeeze().cpu().tolist())
            all_labels.extend(label_chunk.squeeze().cpu().tolist())

        # Accumulate chunked losses and perform optimization step
        if chunk_losses:
            scaler.step(optimizer)  # Update weights
            scaler.update()  # Update the scaler
            total_loss += sum(chunk_losses) / len(chunk_losses)

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")

    return avg_loss, all_probs, all_labels



def validate_or_test_epoch(loader, model, criterion, device, chunk_size, mode="Validation"):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for normal_features, anomalous_features in tqdm(loader, desc=mode):
            normal_features = normal_features.squeeze(-1).squeeze(-1).squeeze(-1).to(device)
            anomalous_features = anomalous_features.squeeze(-1).squeeze(-1).squeeze(-1).to(device)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)

            labels = torch.cat((
                torch.ones(len(normal_features), 1).to(device),
                torch.zeros(len(anomalous_features), 1).to(device)
            ), dim=0)

            # Forward pass with chunks
            outputs = process_in_chunks(inputs, model, chunk_size, device)
            loss = criterion(outputs, labels, batch_size=inputs.size(0))
            total_loss += loss.item()
            all_probs.extend(outputs.squeeze().cpu().tolist())
            all_labels.extend(labels.squeeze().cpu().tolist())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_probs, all_labels


def compute_metrics(all_labels, all_probs, epoch, prefix="train"):
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Save Precision-Recall Curve
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix.capitalize()} Precision-Recall Curve")
    plt.savefig(f"{prefix}_precision_recall_curve_epoch_{epoch}.png")
    plt.close()

    # Save ROC Curve
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix.capitalize()} ROC Curve")
    plt.legend()
    plt.savefig(f"{prefix}_roc_curve_epoch_{epoch}.png")
    plt.close()

    return best_threshold, roc_auc


# Training and Evaluation Loop
num_epochs = 10
chunk_size = 128

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    chunk_size = 128  # Adjust based on memory availability
    train_loss, train_probs, train_labels = train_epoch(train_loader, model, optimizer, criterion, device, scaler, chunk_size)
    train_threshold, train_auc = compute_metrics(train_labels, train_probs, epoch, prefix="train")
    print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")

    val_loss, val_probs, val_labels = validate_or_test_epoch(val_loader, model, criterion, device, chunk_size, mode="Validation")
    val_threshold, val_auc = compute_metrics(val_labels, val_probs, epoch, prefix="val")
    print(f"Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}")

    scheduler.step(val_loss)

# Save Model
torch.save(model.state_dict(), "trained_model.pth")

if __name__ == "__main__":
    import torch
    from extract_segments import extract_segments_from_video  # Function to divide videos into segments
    from extract_features import extract_features  # Function to extract features from segments
    from Sequential_Model import AnomalyAutoencoder  # Your MIL model

    # Define constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_path = r"E:\MIL\Dataset\Normal-Videos\Normal_Videos021_x264.mp4"  # Replace with actual path
    chunk_size = 64  # Adjust chunk size based on memory capacity

    # Load the trained MIL model
    model = AnomalyAutoencoder(input_dim=2048).to(device)
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()

    def infer_video(video_path, model, device, chunk_size):
        """
        Perform inference on a single video to predict its label (0 or 1).

        Args:
            video_path (str): Path to the video file.
            model (torch.nn.Module): Trained MIL model.
            device (torch.device): Device for computation (CPU or GPU).
            chunk_size (int): Number of samples to process at a time.

        Returns:
            float: Predicted probability (0.0 to 1.0) of being anomalous.
        """
        try:
            # Step 1: Extract video segments
            segments = extract_segments_from_video(
                video_path, segment_size=16, target_shape=(240, 320), frame_skip=5
            )
            if not segments:
                raise ValueError("No segments were extracted from the video.")

            # Step 2: Extract features for each segment
            features = []
            with torch.no_grad():
                for segment in segments:
                    segment = segment.unsqueeze(0).to(device)  # Add batch dimension
                    features.append(extract_features(segment, device).cpu())
            features = torch.cat(features, dim=0)  # Combine features

            # Ensure features are reshaped properly
            if features.dim() > 2:
                features = features.squeeze(-1).squeeze(-1).squeeze(-1)
            features = features.view(-1, features.size(-1))

            # Step 3: Process features in chunks
            outputs = []
            for i in range(0, features.size(0), chunk_size):
                chunk = features[i:i + chunk_size].to(device)
                with torch.no_grad():
                    outputs.append(model(chunk))
            scores = torch.cat(outputs, dim=0)

            # Aggregate scores across segments
            predicted_probability = torch.mean(scores).item()
            return predicted_probability

        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    # Perform inference
    predicted_probability = infer_video(video_path, model, device, chunk_size)
    if predicted_probability is not None:
        # Assuming you have the threshold from training
        best_threshold = 0.5  # Replace with the actual threshold used
        predicted_label = 1 if predicted_probability > best_threshold else 0
        print(f"Predicted Probability of being Anomalous: {predicted_probability:.4f}")
        print(f"Predicted Label: {'Anomalous' if predicted_label == 1 else 'Normal'}")
