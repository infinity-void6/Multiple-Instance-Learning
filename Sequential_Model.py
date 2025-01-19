
import torch
import torch.nn as nn
from torch.nn import functional as F

class DeepGRUModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=5, dropout=0.5, use_layer_norm=False):
        super(DeepGRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.use_layer_norm = use_layer_norm
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.fc = nn.Linear(hidden_dim, 1)  # Final classification layer
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        """
        Forward pass for Deep GRU Model
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
        Returns:
            Output tensor of shape [batch_size, 1]
        """
        x = x.float()
        if x.dim() > 2:
            x = x.squeeze(-1).squeeze(-1).squeeze(-1)
            x = x.view(-1, x.size(-1))
            #x = x.unsqueeze(0)
        #print(x.shape)
        # Pass through GRU layers
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, seq_length, hidden_dim]

        #print(gru_out.shape)

        # Optional: Apply layer normalization
        if self.use_layer_norm:
            gru_out = self.layer_norm(gru_out)
        #print(gru_out.shape)

        # Global average pooling over the sequence length
        #pooled_out = gru_out.mean(dim=1)  # [batch_size, hidden_dim]
        #print(pooled_out.shape)

        # Fully connected classification layer
        #gru_out = self.dropout(gru_out)
        out = self.fc(gru_out)  # [batch_size, 1]
        return self.sigmoid(out)

if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load features
    features = torch.load(r'E:\MIL\Code\normal_features\Normal_Videos436_x264_features.pt')['features'].to(device)

    print(f"Input features shape: {features.shape}")
    
    # Initialize model
    model = DeepGRUModel(
            input_dim=2048,  # Feature size
            hidden_dim=256,  # GRU hidden size
            num_layers=16,   # Number of GRU layers
            dropout=0.5,     # Dropout rate
            use_layer_norm=True  # Use layer normalization
        ).to(device)
    model.eval()
    # Forward pass
    with torch.no_grad():
        feature= features.unsqueeze(0).to(device)
        print(f'feature shape:{feature.shape}')
        output = model(features)
    print(f"Output shape: {output.shape}")
    print(f'output:{output}')
    max = 0
    for i in output:
        if i > max:
            max = i
    print(f'max:{max}')
    # print(f"Output scores: {output}")
    # print(f'Max of output:{output.max()}')
    print(f'length of features : {len(features)}')
    print(f'length of output:  {len(output)}')


#------------------------------------------
# class TCNLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, kernel_size, dilation):
#         super(TCNLayer, self).__init__()
#         self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
#         self.dropout = nn.Dropout(0.5)
#         self.norm = nn.BatchNorm1d(hidden_dim)

#     def forward(self, x):
#         # x shape: [batch_size, input_dim, seq_length]
#         out = self.conv1(x)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.dropout(out)
#         out = self.norm(out)
#         return out

# class TCNMILModel(nn.Module):
#     def __init__(self, input_dim=2048, hidden_dim=512, num_layers=3, kernel_size=3):
#         super(TCNMILModel, self).__init__()
#         self.tcn_layers = nn.ModuleList(
#             [TCNLayer(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size, dilation=2 ** i) for i in range(num_layers)]
#         )
#         self.fc = nn.Linear(hidden_dim, 1)  # Final layer for anomaly score prediction
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Ensure input is float
#         x = x.float()

#         # Remove extra dimensions if present
#         if x.dim() > 3:  # Check for extra dimensions [batch_size, seq_length, input_dim, 1, 1]
#             x = x.squeeze(-1).squeeze(-1)  # Remove trailing dimensions
#             print(x.shape)
        
#         # Pass through TCN layers
#         for layer in self.tcn_layers:
#             x = layer(x)

#         # Global pooling over the sequence length
#         x = x.mean(dim=-1)  # [batch_size, hidden_dim]

#         # Final fully connected layer and sigmoid activation
#         x = self.fc(x)  # [batch_size, 1]
#         x = self.sigmoid(x)  # [batch_size, 1]

#         return x
#-----------------------------------------


# ------------------------------------------------------
# class SequentialMILModel(nn.Module):
#     def __init__(self, input_dim=2048, hidden_dim=512):
#         """
#         Initializes a Sequential MIL Model for anomaly detection.

#         Parameters:
#         - input_dim (int): Input feature dimension (default: 2048).
#         - hidden_dim (int): Hidden layer dimension (default: 512).
#         """
#         super(SequentialMILModel, self).__init__()
#         print("SequentialMILModel Initialized")
        
#         self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 32)         # Second fully connected layer
#         self.bn2 = nn.BatchNorm1d(32)
#         self.fc3 = nn.Linear(32, 1)                 # Final output layer
#         self.bn3 = nn.BatchNorm1d(1)
        
#         self.dropout = nn.Dropout(0.6)              # Dropout with probability 0.6
#         self.relu = nn.ReLU()                       # ReLU activation
#         self.sigmoid = nn.Sigmoid()                 # Sigmoid activation for output

#     def forward(self, x):
#         """
#         Forward pass for the model.

#         Parameters:
#         - x (torch.Tensor): Input features of shape [num_segments, input_dim].

#         Returns:
#         - torch.Tensor: Raw logits for each segment. Shape: [num_segments, 1].
#         """
#         x = x.float()
#         if x.dim() > 2:  # Reshape tensor if needed
#             x = x.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
#             x = x.view(-1, x.size(-1))

#         # Fully connected layers with dropout and ReLU
#         x = self.relu(self.fc1(x))
#         x = self.bn1(x)
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.bn2(x)

#         # Final output layer
#         x = self.fc3(x)
#         x = self.bn3(x)
#         x = self.sigmoid(x)

#         return x.squeeze(-1)  # Return anomaly score for each segment
# ----------------------------------------------


#-------------------------------------------
# class GRUMILModel(nn.Module):
#     def __init__(self, input_dim=2048, hidden_dim=512, num_layers=16):
#         super(GRUMILModel, self).__init__()
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = x.float()
#         if x.dim() > 2:
#             x = x.squeeze(-1).squeeze(-1).squeeze(-1)
#             x = x.view(-1, x.size(-1))
#         output, _ = self.gru(x)  # Get the output for each time step
#         segment_scores = self.fc(output)  # Compute a score for each segment
#         return self.sigmoid(segment_scores).squeeze(-1)  # Return anomaly scores for each segment
#-------------------------------------------


#-------------------------------------------
# class SequentialMILModel(nn.Module):
#     def __init__(self, input_dim=2048, hidden_dim1=4096, hidden_dim2=512):
#         super(SequentialMILModel, self).__init__()
#         print("Complex SequentialMILModel Initialized")

#         # Layer configuration: 2048 -> 4096 -> 512 -> 32 -> 1
        
#         self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 2048 -> 4096
#         self.bn1 = nn.BatchNorm1d(hidden_dim1)
        
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 4096 -> 512
#         self.bn2 = nn.BatchNorm1d(hidden_dim2)

#         self.fc3 = nn.Linear(hidden_dim2, 32)         # 512 -> 32
#         self.bn3 = nn.BatchNorm1d(32)

#         self.fc4 = nn.Linear(32, 1)                  # 32 -> 1

#         self.dropout = nn.Dropout(0.6)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
#         self.silu = nn.SiLU()
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x=x.float()
#         if x.dim() > 2:
#             x = x.squeeze(-1).squeeze(-1).squeeze(-1)
#             x = x.view(-1, x.size(-1))

#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.fc2(x)
#         x = self.dropout(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         x = self.fc3(x)
#         x = self.dropout(x)
#         x = self.bn3(x)
#         x = self.relu(x)

#         x = self.fc4(x)
#         x = self.dropout(x)
#         x = self.sigmoid(x)
#         return x.squeeze(-1)
#-----------------------------------------------------

#------------------------------------------------------------------------------------------
# class SequentialMILModel(nn.Module):
#     def __init__(self, input_dim=2048, dropout_rate=0.6):
#         super(SequentialMILModel, self).__init__()
#         print("Updated SequentialMILModel Initialized")

#         # Layer configuration: 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 1
#         self.fc1 = nn.Linear(input_dim, 1024)  # 2048 -> 1024
#         self.bn1 = nn.BatchNorm1d(1024) if 1024 > 1 else None

#         self.fc2 = nn.Linear(1024, 512)         # 1024 -> 512
#         self.bn2 = nn.BatchNorm1d(512)

#         self.fc3 = nn.Linear(512, 256)          # 512 -> 256
#         self.bn3 = nn.BatchNorm1d(256)

#         self.fc4 = nn.Linear(256, 128)          # 256 -> 128
#         self.bn4 = nn.BatchNorm1d(128)

#         self.fc5 = nn.Linearx(128, 64)           # 128 -> 64
#         self.bn5 = nn.BatchNorm1d(64)

#         self.fc6 = nn.Linear(64, 32)            # 64 -> 32
#         self.bn6 = nn.BatchNorm1d(32)

#         self.fc7 = nn.Linear(32, 1)             # 32 -> 1

#         self.dropout = nn.Dropout(dropout_rate)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
#         self.sigmoid = nn.Sigmoid()
#         self.swish = nn.SiLU()
#         self.relu = nn.ReLU()


#     def forward(self, x):
#         # Handle multi-dimensional inputs
#         if x.dim() > 2:
#             x = x.squeeze(-1).squeeze(-1).squeeze(-1)
#             x = x.view(-1, x.size(-1))

#         # First fully connected layer
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         # Second fully connected layer
#         x = self.fc2(x)
#         #x = self.bn2(x)
#         x = self.relu(x)

#         # Third fully connected layer
#         x = self.fc3(x)
#         #x = self.bn3(x)
#         x = self.relu(x)

#         # Fourth fully connected layer
#         x = self.fc4(x)
#         #x = self.bn4(x)
#         x = self.relu(x)

#         # Fifth fully connected layer
#         x = self.fc5(x)
#         x = self.bn5(x)
#         x = self.relu(x)

#         # Sixth fully connected layer
#         x = self.fc6(x)
#         x = self.bn6(x)
#         x = self.relu(x)

#         # Final layer
#         x = self.fc7(x)
#         x = self.sigmoid(x)  # Output in range [0, 1]

#         return x.squeeze(-1)
# -----------------------------------------------------------------









#------------------------------------------------------------------
# import torch
# import torch.nn as nn

# class AttentionMILModel(nn.Module):
#     def __init__(self, input_dim=2048):
#         super(AttentionMILModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x=x.float()
#         if x.dim() > 2:
#              x = x.squeeze(-1).squeeze(-1).squeeze(-1)
#              x = x.view(-1, x.size(-1))
#         x = self.fc1(x)
#         x, _ = self.attention(x, x, x)  # Self-attention
#         x = nn.ReLU()(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return self.sigmoid(x).squeeze(-1)
# -----------------------------------------------------------------

# ------------------------------------------------------------------
# class GRUMILModel(nn.Module):
#     def __init__(self, input_dim=2048, hidden_dim=512, num_layers=2):
#         super(GRUMILModel, self).__init__()
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = x.float()
#         if x.dim() > 2:
#             x = x.squeeze(-1).squeeze(-1).squeeze(-1)
#             x = x.view(-1, x.size(-1))
#         output, _ = self.gru(x)  # Get the output for each time step
#         segment_scores = self.fc(output)  # Compute a score for each segment
#         return self.sigmoid(segment_scores).squeeze(-1)  # Return anomaly scores for each segment
#----------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------
# class SequentialMILModel(nn.Module):
#     def __init__(self, input_dim=2048, hidden_dim=512):
#         super(SequentialMILModel, self).__init__()
#         print("Complex SequentialMILModel Initialized")

#         # Layer configuration: 2048 -> 512 -> 32 -> 1
        
#         self.fc1 = nn.Linear(input_dim, hidden_dim)  # 2048 -> 512
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
        
#         self.fc2 = nn.Linear(hidden_dim, 32)         # 512 -> 32
#         self.bn2 = nn.BatchNorm1d(32)

#         self.fc3 = nn.Linear(32, 1)                 # 32 -> 1

#         self.dropout = nn.Dropout(0.4)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
#         self.silu = nn.SiLU()
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         if x.dim() > 2:
#             x = x.squeeze(-1).squeeze(-1).squeeze(-1)
#             x = x.view(-1, x.size(-1))

#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
   
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.sigmoid(self.fc3(x))
#         x = self.dropout(x)

#         return x.squeeze(-1)

    # def __init__(self, input_dim=2048, hidden_dim=1024):
    #     super(SequentialMILModel, self).__init__()
    #     print("Complex SequentialMILModel Initialized")

    #     #1024,512,256,128,64,32,1
        
    #     self.fc1 = nn.Linear(input_dim, hidden_dim) #2048 - 1024
    #     self.bn1 = nn.BatchNorm1d(hidden_dim)
        
    #     self.fc2 = nn.Linear(hidden_dim, 512) #1024 - 512
    #     self.bn2 = nn.BatchNorm1d(512)

    #     self.fc3 = nn.Linear(512, 256) # 512 -256
    #     self.bn3 = nn.BatchNorm1d(256)

    #     self.fc4 = nn.Linear(256, 128)
    #     self.bn4=nn.BatchNorm1d(128)

    #     self.fc5 = nn.Linear(128,64)
    #     self.bn5=nn.BatchNorm1d(64)

    #     self.fc6 =nn.Linear(64,32)

    #     self.fc7 = nn.Linear(32,1)
        
        
    #     self.dropout = nn.Dropout(0.4)
    #     self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    #     self.silu = nn.SiLU()
    #     self.relu = nn.ReLU()
    #     self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     if x.dim() > 2:
    #         x = x.squeeze(-1).squeeze(-1).squeeze(-1)
    #         x = x.view(-1, x.size(-1))

    #     x = self.fc1(x)
    #     x = self.relu(x)

    #     x = self.fc2(x)
    #     x = self.relu(x)
        
    #     x = self.fc3(x)
    #     x = self.relu(x)

    #     x = self.fc4(x)
    #     x = self.relu(x)

    #     x = self.fc5(x)
    #     x = self.relu(x)

    #     x = self.fc6(x)
    #     x = self.relu(x)

    #     x = self.fc7(x)
    #     x = self.sigmoid(x)

    #     return x.squeeze(-1)
    #-------------------------------------------------------------------------

