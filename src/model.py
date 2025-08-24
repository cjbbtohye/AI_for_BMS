# src/model.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, AdaptiveAvgPool1d

# --- CNN-LSTM Model ---
class CNNLSTM(nn.Module):
    """CNN-LSTM model for SOH prediction from cycle timeseries data."""
    def __init__(self, input_features, cnn_out_channels, kernel_size, lstm_hidden_size, lstm_num_layers, sequence_length):
        super(CNNLSTM, self).__init__()
        self.cnn_out_channels = cnn_out_channels

        # CNN layers to extract features from each cycle
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=cnn_out_channels, kernel_size=kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layer to process the sequence of features from past cycles
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=False  # LSTM expects (seq_len, batch, features)
        )

        # Fully connected layer to predict a single SOH value
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        # Input x shape: (batch_size, look_back_cycles, input_features, sequence_length)
        batch_size, look_back_cycles, input_features, sequence_length = x.size()

        # Reshape for CNN: Combine batch and look_back dimensions
        x = x.view(batch_size * look_back_cycles, input_features, sequence_length)

        # --- CNN Feature Extraction ---
        x = self.pool1(self.relu1(self.conv1(x)))

        # Global Average Pooling to get a fixed-size feature vector for each cycle
        x = AdaptiveAvgPool1d(1)(x)
        x = x.view(batch_size * look_back_cycles, self.cnn_out_channels)

        # Reshape for LSTM: (batch * look_back, features) -> (look_back, batch, features)
        x = x.view(batch_size, look_back_cycles, self.cnn_out_channels).permute(1, 0, 2)

        # --- LSTM Processing ---
        # lstm_out contains the output for each timestep
        # h_n is the final hidden state
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the hidden state of the last layer for the final prediction
        last_hidden_state = h_n[-1]

        # --- Fully Connected Layer ---
        out = self.fc(last_hidden_state)
        return out

# --- CNN-Transformer Model ---
class CNNTransformer(nn.Module):
    """CNN-Transformer model for SOH prediction from cycle timeseries data."""
    def __init__(self, input_features, cnn_out_channels, kernel_size,
                 transformer_nhead, transformer_num_encoder_layers, transformer_dim_feedforward,
                 sequence_length, dropout=0.1):
        super(CNNTransformer, self).__init__()
        self.cnn_out_channels = cnn_out_channels

        # CNN layers to extract features from each cycle
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=cnn_out_channels, kernel_size=kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Transformer Encoder Layer definition
        encoder_layers = TransformerEncoderLayer(
            d_model=cnn_out_channels,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            batch_first=True  # Transformer expects (batch, seq_len, features)
        )
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=transformer_num_encoder_layers
        )

        # Fully connected layer to predict a single SOH value
        self.fc = nn.Linear(cnn_out_channels, 1)

    def forward(self, x):
        # Input x shape: (batch_size, look_back_cycles, input_features, sequence_length)
        batch_size, look_back_cycles, input_features, sequence_length = x.size()

        # Reshape for CNN: Combine batch and look_back dimensions
        x = x.view(batch_size * look_back_cycles, input_features, sequence_length)

        # --- CNN Feature Extraction ---
        x = self.pool1(self.relu1(self.conv1(x)))

        # Reshape for Transformer: (batch * look_back, features, seq_len) -> (batch * look_back, seq_len, features)
        x = x.permute(0, 2, 1)

        # --- Transformer Encoder ---
        transformer_out = self.transformer_encoder(x)

        # Aggregate Transformer output (mean pooling over the sequence dimension)
        aggregated_out = transformer_out.mean(dim=1)

        # Reshape back to separate batch and look_back cycles
        aggregated_out = aggregated_out.view(batch_size, look_back_cycles, self.cnn_out_channels)

        # Aggregate features across the look_back window (e.g., by averaging)
        final_representation = aggregated_out.mean(dim=1)

        # --- Fully Connected Layer ---
        out = self.fc(final_representation)
        return out