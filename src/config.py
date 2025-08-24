# src/config.py
import torch

# --- Directories and Paths ---
# All paths are relative to the project root directory
NPZ_DIR_PATH = './data/dataset_out/'
FIG_SAVE_DIR = './results/figures/'
REPORT_SAVE_DIR = './results/reports/'
MODEL_SAVE_DIR = './models/'
PRE_TRAINED_MODEL_NAME = "best_model_cnntransformer.pth"

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Selection ---
MODEL_TYPE = 'CNNTransformer' # Options: 'CNNLSTM', 'CNNTransformer'

# --- Data Parameters ---
NUM_POINTS_PER_CYCLE = 1000      # Number of data points per cycle
INPUT_FEATURES = 3               # Number of features to use (e.g., Voltage, Capacity, dQ/dV)
FEATURE_INDICES = [2, 3, 5]      # Column indices for Voltage, Capacity, dQ/dV
LOOK_BACK_CYCLES = 5             # Number of previous cycles to use as input

# --- Model Hyperparameters ---
# Shared
CNN_OUT_CHANNELS = 16
CNN_KERNEL_SIZE = 5
# LSTM specific
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 1
# Transformer specific
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_ENCODER_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 128
TRANSFORMER_DROPOUT = 0.1

# --- Training Hyperparameters ---
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
# SOH threshold to determine the starting point for prediction during evaluation phase of training
SOH_PRED_START_THRESH = 0.95

# --- Evaluation & Prediction Parameters ---
# EOL (End of Life) definition
EOL_THRESHOLD = 0.8
# SOH threshold for selecting data for fine-tuning
FINE_TUNE_SOH_THRESHOLD = 0.8
# Number of epochs for fine-tuning
FINE_TUNE_EPOCHS = 2
# Learning rate for fine-tuning
FINE_TUNE_LR = 1e-4