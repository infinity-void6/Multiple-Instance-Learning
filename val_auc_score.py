import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from Sequential_Model import DeepGRUModel
from split_data_train_val_test import create_dataloader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# Initialize the computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = DeepGRUModel(
    input_dim=2048,  # Feature size
    hidden_dim=256,  # GRU hidden size
    num_layers=16,   # Number of GRU layers
    dropout=0.5,     # Dropout rate
    use_layer_norm=True  # Use layer normalization
).to(device)

# Load the saved weights
weights_path = r"E:\MIL\Code\Metrics\best_model_val_roc_auc_0.6860.pth"
checkpoint = torch.load(weights_path, map_location=device)

# Ensure weights are loaded correctly
if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.eval()  # Set the model to evaluation mode

# Loss function
def ranking_loss_val(scores, labels, batch_size, lamda_sparsity=8e-5, lamda_smooth=8e-5):
    num_segments = scores.shape[0] // batch_size
    total_loss = torch.tensor(0.0, requires_grad=True, device=scores.device)

    for i in range(batch_size):
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i].float()

        mean_score = video_scores.mean()
        bce_loss = F.binary_cross_entropy_with_logits(mean_score, video_label)

        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))
        smoothness_loss = lamda_smooth * torch.sum(
            (torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2
        )

        total_loss = total_loss + bce_loss + sparsity_loss + smoothness_loss

    return total_loss / batch_size

criterion_val = ranking_loss_val

# Validation Data Loader
val_normal_dir = "split/val/normal"
val_anomalous_dir = "split/val/anomalous"
max_segments_val = 253
batch_size = 4

val_loader = create_dataloader(val_normal_dir, val_anomalous_dir, batch_size, max_segments_val)

# Standard scaler for feature scaling
StandardScale = StandardScaler()

# Validate the model
all_probs = []
all_labels = []
total_loss = 0.0

with torch.no_grad():
    for normal_features, anomalous_features in tqdm(val_loader, desc="Validation"):
        # Scale and reshape normal features
        if normal_features.dim() > 2:
            normal_features = normal_features.squeeze(-1).squeeze(-1).view(-1, 2048)
            normal_features = torch.tensor(
                StandardScale.fit_transform(normal_features.cpu().numpy()),
                dtype=torch.float32
            ).to(device)

        # Scale and reshape anomalous features
        if anomalous_features.dim() > 2:
            anomalous_features = anomalous_features.squeeze(-1).squeeze(-1).view(-1, 2048)
            anomalous_features = torch.tensor(
                StandardScale.fit_transform(anomalous_features.cpu().numpy()),
                dtype=torch.float32
            ).to(device)

        # Prepare inputs and labels
        normal_features = normal_features.unsqueeze(2).to(device)
        anomalous_features = anomalous_features.unsqueeze(2).to(device)
        inputs = torch.cat((normal_features, anomalous_features), dim=0)

        labels = torch.cat((
            torch.ones(len(normal_features)).to(device),
            torch.zeros(len(anomalous_features)).to(device)
        ), dim=0)

        # Forward pass
        outputs = model(inputs)
        loss = criterion_val(outputs, labels, batch_size)
        total_loss += loss.item()

        # Collect probabilities and labels
        all_probs.extend(torch.sigmoid(outputs).view(-1).cpu().tolist())
        all_labels.extend(labels.view(-1).cpu().tolist())

# Compute AUC score
auc_score = roc_auc_score(all_labels, all_probs)
print(f"Validation AUC: {auc_score:.4f}")
