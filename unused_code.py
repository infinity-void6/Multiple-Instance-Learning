# epoch_training_test.py

import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from split_data_train_val_test import create_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve,f1_score

# Assuming SequentialMILModel is already implemented
from Sequential_Model import SequentialMILModel
from loss_function import ranking_loss

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize Model
model = SequentialMILModel(input_dim=2048).to(device)

# 2. Define Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Scheduler: Reduces learning rate when validation loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

# 3. Define Loss Function
criterion = ranking_loss  # Replace with your ranking loss function

# 4. Gradient Scaler for Mixed Precision
scaler = GradScaler()

# Data Loaders
# Directories for train, val, and test splits
train_normal_dir = "split/train/normal"
train_anomalous_dir = "split/train/anomalous"
val_normal_dir = "split/val/normal"
val_anomalous_dir = "split/val/anomalous"
test_normal_dir = "split/test/normal"
test_anomalous_dir = "split/test/anomalous"

# Max segments for each split
max_segments_train = 887  # Max for training
max_segments_val = 675    # Max for validation
max_segments_test = 271   # Max for testing

# Batch size
batch_size = 4

# Create DataLoaders
train_loader = create_dataloader(train_normal_dir, train_anomalous_dir, batch_size, max_segments_train)
val_loader = create_dataloader(val_normal_dir, val_anomalous_dir, batch_size, max_segments_val)
test_loader=create_dataloader(test_normal_dir,test_anomalous_dir,batch_size, max_segments_test)

'''def print_model_weights(model, epoch):
    """
    Prints the weights of the model's layers.
    
    Args:
        model (torch.nn.Module): Trained model.
        epoch (int): Current epoch number.
    """
    print(f"\nWeights after Epoch {epoch + 1}:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}")
            print(f"Weight: {param.data}")
            print(f"Weight Shape: {param.data.shape}")
            print(f"Weight Mean: {param.data.mean():.6f}, Weight Std: {param.data.std():.6f}\n")

# Training Loop
def train_epoch(train_loader, model, optimizer, criterion, device, scaler, batch_size):
    model.train()  # Set model to training mode
    total_loss = 0.0

    for normal_features, anomalous_features in tqdm(train_loader, desc="Training"):
        # Combine features
        if normal_features.dim() > 2:            
            normal_features = normal_features.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
            normal_features = normal_features.view(-1, normal_features.size(-1))          # Reshape to [n * pad seg, 2048]
            #print(f"Normal features reshaped: {normal_features.shape}")
        if anomalous_features.dim() > 2:
            anomalous_features=anomalous_features.squeeze(-1).squeeze(-1).squeeze(-1)
            anomalous_features = anomalous_features.view(-1,anomalous_features.size(-1))
            #print(f"Anomalous features reshaped: {anomalous_features.shape}")
        
        normal_features = normal_features.to(device)
        anomalous_features = anomalous_features.to(device)
        inputs = torch.cat((normal_features, anomalous_features), dim=0)

        # Labels: 1 for normal, 0 for anomalous
        labels = torch.cat((
            torch.ones(len(normal_features), 1).to(device),  # Normal
            torch.zeros(len(anomalous_features), 1).to(device)  # Anomalous
        ), dim=0)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, labels, batch_size)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

# Validation Loop
def validate_epoch(val_loader, model, criterion, device, batch_size):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # No gradient calculation
        for normal_features, anomalous_features in tqdm(val_loader, desc="Validation"):
            # Combine features
            if normal_features.dim() > 2:            
                normal_features = normal_features.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
                normal_features = normal_features.view(-1, normal_features.size(-1))   # Reshape to [n * pad seg, 2048]
                #print(f"Normal features reshaped: {normal_features.shape}")
            if anomalous_features.dim() > 2:
                anomalous_features=anomalous_features.squeeze(-1).squeeze(-1).squeeze(-1)
                anomalous_features = anomalous_features.view(-1,anomalous_features.size(-1))
                #print(f"Anomalous features reshaped: {anomalous_features.shape}")
            
            normal_features = normal_features.to(device)
            anomalous_features = anomalous_features.to(device)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)

            # Labels: 1 for normal, 0 for anomalous
            labels = torch.cat((
                torch.ones(len(normal_features), 1).to(device),  # Normal
                torch.zeros(len(anomalous_features), 1).to(device)  # Anomalous
            ), dim=0)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels,batch_size)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def test_epoch(test_loader, model, criterion, device, batch_size):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    all_labels = []
    all_probs = []
    with torch.no_grad():  # No gradient calculation
        for normal_features, anomalous_features in tqdm(test_loader, desc="Validation"):
            # Combine features
            if normal_features.dim() > 2:            
                normal_features = normal_features.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
                normal_features = normal_features.view(-1, normal_features.size(-1))   # Reshape to [n * pad seg, 2048]
                #print(f"Normal features reshaped: {normal_features.shape}")
            if anomalous_features.dim() > 2:
                anomalous_features=anomalous_features.squeeze(-1).squeeze(-1).squeeze(-1)
                anomalous_features = anomalous_features.view(-1,anomalous_features.size(-1))
                #print(f"Anomalous features reshaped: {anomalous_features.shape}")
            
            normal_features = normal_features.to(device)
            anomalous_features = anomalous_features.to(device)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)

            # Labels: 1 for normal, 0 for anomalous
            labels = torch.cat((
                torch.ones(len(normal_features), 1).to(device),  # Normal
                torch.zeros(len(anomalous_features), 1).to(device)  # Anomalous
            ), dim=0)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels,batch_size)
            total_loss += loss.item()
                       # Collect probabilities and labels
            all_probs.extend(outputs.squeeze().cpu().tolist())
            all_labels.extend(labels.squeeze().cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

    # Compute optimal threshold
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    print(f"Best Threshold: {best_threshold:.4f}, Best F1 Score: {f1_scores[best_threshold_index]:.4f}")

    return avg_loss, best_threshold

num_epochs = 10
batch_size = 4  # Keep consistent with DataLoader



for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(train_loader, model, optimizer, criterion, device, scaler, batch_size)
    val_loss,best_threshold = validate_epoch(val_loader, model, criterion, device, batch_size)
    test_loss=test_epoch(test_loader,model, criterion, device, batch_size)
    scheduler.step(val_loss)
    
    # Print model weights
    # print_model_weights(model, epoch)

torch.save(model.state_dict(), "trained_model.pth")
'''
if __name__ == "__main__":
    import torch
    import cv2
    from extract_segments import extract_segments_from_video  # Function to divide videos into segments
    from extract_features import extract_features  # Function to extract features from segments
    from Sequential_Model import SequentialMILModel  # Your MIL model
    device=torch.device('cuda')
    best_train_threshold=9.5367431640625e-07
    best_threshold= 0.00028432568069547415

    def infer_video(video_path, model, device):
        """
        Perform inference on a single video to predict its label (0 or 1).

        Args:
            video_path (str): Path to the video file.
            model (torch.nn.Module): Trained MIL model.
            device (torch.device): Device for computation (CPU or GPU).

        Returns:
            float: Predicted probability (0.0 to 1.0) of being anomalous.
        """
        try:
            # Step 1: Extract video segments
            segments = extract_segments_from_video(video_path, segment_size=16, target_shape=(240, 320), frame_skip=5)
            if not segments:
                raise ValueError("No segments were extracted from the video.")

            # Step 2: Extract features for each segment
            model.eval()
            features = []
            with torch.no_grad():
                for segment in segments:
                    segment = segment.unsqueeze(0).to(device)  # Add batch dimension
                    segment_features = extract_features(segment, device)
            
            # Combine features from all segments
            segment_features = torch.cat(segment_features, dim=0).to(device)  # Shape: [num_segments, feature_dim]
            if segment_features.dim() > 2:
                segment_features=segment_features.squeeze(-1).squeeze(-1).squeeze(-1)
                segment_features = segment_features.view(-1,segment_features.size(-1))

            # Step 3: Predict using the trained MIL model
            with torch.no_grad():
                scores = model(segment_features)
                predicted_probability = torch.mean(scores).item()  # Aggregate scores across segments
            
            return predicted_probability

        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    # File path for the video
    video_path = r'E:\MIL\Code\videos\walking.mp4'  # Replace with the actual path to the new video

    # Device setup
    device = torch.device('cuda')

    # Load the trained MIL model
    model = SequentialMILModel(input_dim=2048).to(device)
    model.load_state_dict(torch.load("trained_model.pth"))  # Replace with the actual path to your model weights

    # Perform inference
    predicted_probability = infer_video(video_path, model, device)
    if predicted_probability is not None:
        predicted_label = 0 if predicted_probability > best_threshold else 1
        print(f"Predicted Probability of being Anomalous: {predicted_probability:.4f}")
        print(f"Predicted Label: {predicted_label}")


'''--------
from Sequential_Model import SequentialMILModel
import torch
from torch import nn
from split_data_train_val_test import DataPreparation
from loss_function import combined_loss
from torch.amp import autocast 
# Automatically switches between float16 and float32 precision during 
# training for faster computation.
from torch.amp import GradScaler
# Scales gradients to prevent underflow when using float16. not go to 0
# Initialize Model, Optimizer, and Scaler
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SequentialMILModel(input_dim=2048).to(device)
# Initialize optimizer and scheduler
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",         # Minimize the monitored metric (e.g., loss)
    factor=0.1,         # Reduce LR by a factor of 0.1
    patience=5,         # Wait 5 epochs for improvement before reducing LR
    verbose=True        # Print updates when LR changes
)
scaler = GradScaler()  # Gradient scaler for mixed precision

# Path to the directory containing processed feature files
processed_features_dir = "processed_features"

# Initialize the DataPreparation class
data_prep = DataPreparation(processed_features_dir)
# Split the dataset
data_prep.split_dataset(test_size=0.4, val_size=0.5)
# Create DataLoaders
train_loader, val_loader,test_loader = data_prep.get_data_loaders(batch_size=1)


# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0

    # Training Progress Bar
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training", leave=False)
    ce_loss=0
    sparsity_loss=0
    smoothness_loss=0
    for features, label in train_progress:
        features = features.squeeze(0).to(device)  # [num_segments, feature_dim] [Segments, 2048]
        label = label.to(device)  # [1]

        optimizer.zero_grad()  # Reset gradients

        # Forward pass with autocast for mixed precision
        with autocast(device_type='cuda', dtype=torch.float32):
            scores = model(features)  # Get raw logits from the model
            scores=torch.mean(scores).unsqueeze(0)
            loss = combined_loss(scores, label) 


        # Backward pass with gradient scaling
        scaler.scale(loss).backward()  # Scale gradients for mixed precision
        scaler.step(optimizer)  # Update model parameters
        scaler.update()  # Update gradient scaler for next step

        train_loss += loss.item()  # Accumulate training loss


        # Update training progress bar with current loss
        train_progress.set_postfix(loss=loss.item())
        ce_loss+= nn.BCEWithLogitsLoss()(scores, label)
        sparsity_loss+= 1e-6 * torch.sum(torch.sigmoid(scores))
        smoothness_loss+= 1e-6 * torch.sum((torch.sigmoid(scores[1:]) - torch.sigmoid(scores[:-1])) ** 2)
    print(f"BCE={ce_loss.item()}, Sparsity={sparsity_loss.item()}, Smoothness={smoothness_loss.item()}")
    print(f'train loss - epoch:{train_loss} - {epoch}')
    
print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")'''


'''
    # Validation Loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation", leave=False)
    with torch.no_grad():  # Disable gradient calculations during validation
        i=0
        for features, label in val_progress:
            
            features = features.squeeze(0).to(device).to(dtype=torch.float32)  # Convert to float32
            label = label.to(device)

            # Forward pass
            scores = model(features)
            scores=torch.mean(scores).unsqueeze(0)
            loss = combined_loss(scores, label)  # Calculate validation loss
            print(f'validation_loss at iteration {i} is : {loss.item()}')
            val_loss += loss.item()
            i+=1
            print(i)

            # Update validation progress bar with current loss
            val_progress.set_postfix(loss=loss.item())
'''
    # Print epoch summary
    # print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")
    # print(f"Val Loss: {val_loss:.4f}")

'''
# Test Loop
model.eval()
test_scores = []
test_labels = []
test_progress = tqdm(test_loader, desc="Testing")

with torch.no_grad():
    for features, label in test_progress:
        features = features.squeeze(0).to(device).to(dtype=torch.float32)
        label = label.to(device)
        scores = model(features)
        scores=torch.mean(scores).unsqueeze(0)
        test_scores.append(scores.item())
        test_labels.append(label.item())

# Calculate Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score

# print(test_scores)

predictions = [1 if score > 0.5 else 0 for score in test_scores]
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)

print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
'''

#loss_function.py

'''import torch
import torch.nn.functional as F

def ranking_loss(scores, labels, batch_size, lamda_sparsity=1e-6, lamda_smooth=1e-6, margin=0.5):
    """
    Ranking loss for weakly-supervised MIL anomaly detection.

    Parameters:
    - scores (torch.Tensor): Predicted scores for all segments. Shape: [batch_size * num_segments].
    - labels (torch.Tensor): Binary labels for videos. Shape: [batch_size].
    - batch_size (int): Number of videos per batch.
    - lamda_sparsity (float): Weight for sparsity loss. Default: 1e-6.
    - lamda_smooth (float): Weight for smoothness loss. Default: 1e-6.
    - margin (float): Margin for ranking loss. Default: 1.0.

    Returns:
    - torch.Tensor: The combined ranking loss.
    """
    num_segments = scores.shape[0] // batch_size  # Segments per video
    total_loss = 0.0  # Initialize cumulative loss

    for i in range(batch_size):
        # Extract scores for the current video
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i]
        # print(f'video_label:{video_label}')

        # Compute max scores for anomaly and normal cases
        max_anomalous = torch.tensor(float("-inf"), device=scores.device)
        max_normal = torch.tensor(float("-inf"), device=scores.device)

        if video_label == 0:  # Anomalous video
            max_anomalous = torch.max(video_scores)
            #print(f'max_anomalous:{max_anomalous}')
        elif video_label == 1:  # Normal video
            max_normal = torch.max(video_scores)
            #print(f'max_normal:{max_normal}')

        # Compute ranking loss: Ensuring valid conditions
        if max_anomalous != float("-inf") and max_normal != float("-inf"):
            rank_loss = F.relu(margin - max_anomalous + max_normal)
            total_loss += rank_loss

        # Sparsity loss: Encourage sparsity in scores
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))

        # Smoothness loss: Penalize abrupt changes between adjacent segments
        smoothness_loss = lamda_smooth * torch.sum(
            (torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2
        )

        total_loss += sparsity_loss + smoothness_loss

    # Normalize by batch size
    return total_loss / batch_size'''


'''import torch
import torch.nn.functional as F

def ranking_loss(scores, labels, batch_size, lamda_sparsity=1e-6, lamda_smooth=1e-6, margin=1.0):
    """
    Ranking loss for weakly-supervised MIL anomaly detection.

    Parameters:
    - scores (torch.Tensor): Predicted scores for all segments. Shape: [batch_size * num_segments].
    - labels (torch.Tensor): Binary labels for videos. Shape: [batch_size].
    - batch_size (int): Number of videos per batch.
    - lamda_sparsity (float): Weight for sparsity loss. Default: 1e-6.
    - lamda_smooth (float): Weight for smoothness loss. Default: 1e-6.
    - margin (float): Margin for ranking loss. Default: 1.0.

    Returns:
    - torch.Tensor: The combined ranking loss.
    """
    num_segments = scores.shape[0] // batch_size  # Assume all videos have the same number of segments
    loss = torch.tensor(0.0, device=scores.device, requires_grad=True)  # Initialize loss

    for i in range(batch_size):
        # Extract scores for the current video
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i]

        # Compute the max score for the video
        max_score = torch.max(video_scores)

        if video_label == 0:  # Anomalous video
            max_anomalous = max_score
        elif video_label == 1:  # Normal video
            max_normal = max_score

        # Ranking loss: max_anomalous > max_normal + margin
        ranking_loss = F.relu(margin - max_anomalous + max_normal)
        loss += ranking_loss

        # Add sparsity and smoothness losses
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))
        smoothness_loss = lamda_smooth * torch.sum((torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2)

        loss += sparsity_loss + smoothness_loss

    # Normalize by batch size
    return loss / batch_size
'''

'''import torch
from torch import nn
def combined_loss(scores, labels, lamda_sparsity=1e-6, lamda_smooth=1e-6):
    """
    Calculates the combined loss function for weakly-supervised MIL anomaly detection.

    The combined loss consists of:
    1. Binary Cross-Entropy Loss with Logits:
       - This is the classification loss that measures how well the model predicts the 
         anomaly scores for each segment. It internally applies the sigmoid function 
         to convert logits into probabilities before calculating the loss.
    
    2. Sparsity Loss:
       - This encourages sparsity in anomaly predictions by penalizing the sum of 
         predicted probabilities across all segments. It helps the model focus on 
         a small number of segments likely to contain anomalies.

    3. Smoothness Loss:
       - This penalizes abrupt changes in predictions across adjacent segments 
         to ensure temporal smoothness in the predicted anomaly scores.

    Parameters:
    - scores (torch.Tensor): Predicted logits from the model. Shape: [num_segments].
    - labels (torch.Tensor): Ground truth labels. Shape: [num_segments].
    - lamda_sparsity (float): Weight for sparsity loss. Default: 8e-5.
    - lamda_smooth (float): Weight for smoothness loss. Default: 8e-5.

    Returns:
    - torch.Tensor: The combined loss value.
    """
    # Binary Cross-Entropy Loss with Logits (handles sigmoid internally)
    #ce_loss = nn.BCEWithLogitsLoss()(scores, labels)

    # Sparsity Loss: Penalizes the sum of predicted probabilities (encourages sparse anomaly detection)
    # sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(scores))

    # Smoothness Loss: Penalizes abrupt changes in adjacent segment predictions
    # smoothness_loss = lamda_smooth * torch.sum((torch.sigmoid(scores[1:]) - torch.sigmoid(scores[:-1])) ** 2)

    # Combined loss
    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([1.5]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_loss = bce_loss(scores, labels)
    sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(scores))
    smoothness_loss = lamda_smooth * torch.sum((torch.sigmoid(scores[1:]) - torch.sigmoid(scores[:-1])) ** 2)

    

    return ce_loss + sparsity_loss + smoothness_loss'''