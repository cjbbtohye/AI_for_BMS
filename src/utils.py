# src/utils.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Import model classes and config to be used in prediction function
from src.model import CNNLSTM, CNNTransformer
import src.config as cfg

def predict_soh_until_eol(model_path, all_timeseries_data, soh_map, start_cycle, fine_tune=True):
    """
    Predicts SOH cycle by cycle, with an option to fine-tune the model first.
    """
    # --- 1. Load Model ---
    print(f"\n--- Loading model from: {model_path} ---")
    try:
        if cfg.MODEL_TYPE == 'CNNLSTM':
             model = CNNLSTM(input_features=cfg.INPUT_FEATURES, cnn_out_channels=cfg.CNN_OUT_CHANNELS, kernel_size=cfg.CNN_KERNEL_SIZE, lstm_hidden_size=cfg.LSTM_HIDDEN_SIZE, lstm_num_layers=cfg.LSTM_NUM_LAYERS, sequence_length=cfg.NUM_POINTS_PER_CYCLE)
        elif cfg.MODEL_TYPE == 'CNNTransformer':
             model = CNNTransformer(input_features=cfg.INPUT_FEATURES, cnn_out_channels=cfg.CNN_OUT_CHANNELS, kernel_size=cfg.CNN_KERNEL_SIZE, transformer_nhead=cfg.TRANSFORMER_NHEAD, transformer_num_encoder_layers=cfg.TRANSFORMER_NUM_ENCODER_LAYERS, transformer_dim_feedforward=cfg.TRANSFORMER_DIM_FEEDFORWARD, sequence_length=cfg.NUM_POINTS_PER_CYCLE, dropout=cfg.TRANSFORMER_DROPOUT)
        else:
             raise ValueError(f"Unknown MODEL_TYPE: {cfg.MODEL_TYPE}")

        # --- FIX: Add weights_only=True for security and to remove the warning ---
        state_dict = torch.load(model_path, map_location=cfg.DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(cfg.DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}

    # --- 2. Fine-tuning (Optional) ---
    if fine_tune:
        print(f"\n--- Preparing data for fine-tuning ({cfg.FINE_TUNE_EPOCHS} epochs) ---")
        ft_features_list, ft_targets_list = [], []
        for target_cycle in sorted(soh_map.keys()):
            if soh_map[target_cycle] < cfg.FINE_TUNE_SOH_THRESHOLD: continue
            input_first_cycle = target_cycle - cfg.LOOK_BACK_CYCLES
            if input_first_cycle < 0: continue
            if (target_cycle * cfg.NUM_POINTS_PER_CYCLE) > all_timeseries_data.shape[0]: continue

            cycle_features_inner = []
            valid_lookback = True
            for c_idx in range(input_first_cycle, target_cycle):
                ft_start = c_idx * cfg.NUM_POINTS_PER_CYCLE
                ft_end = ft_start + cfg.NUM_POINTS_PER_CYCLE
                if ft_end > all_timeseries_data.shape[0]: valid_lookback = False; break
                cycle_data = all_timeseries_data[ft_start:ft_end, cfg.FEATURE_INDICES]
                if not np.all(np.isfinite(cycle_data)): valid_lookback = False; break
                cycle_features_inner.append(torch.tensor(cycle_data, dtype=torch.float32).permute(1, 0))

            if valid_lookback and cycle_features_inner:
                ft_features_list.append(torch.stack(cycle_features_inner, dim=0))
                ft_targets_list.append(torch.tensor([soh_map[target_cycle]], dtype=torch.float32))

        if ft_features_list:
            fine_tune_features = torch.stack(ft_features_list, dim=0).to(cfg.DEVICE)
            fine_tune_targets = torch.stack(ft_targets_list, dim=0).to(cfg.DEVICE)
            model.train()
            criterion = nn.MSELoss().to(cfg.DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=cfg.FINE_TUNE_LR)
            print(f"Starting fine-tuning with {fine_tune_features.shape[0]} samples...")
            for epoch in range(cfg.FINE_TUNE_EPOCHS):
                optimizer.zero_grad()
                predictions = model(fine_tune_features)
                loss = criterion(predictions, fine_tune_targets)
                if torch.isnan(loss): break
                loss.backward()
                optimizer.step()
            print("Fine-tuning finished.")
        else:
            print("Warning: No valid data for fine-tuning. Skipping.")

    # --- 3. Prediction ---
    model.eval()
    predicted_soh = {}
    current_cycle = start_cycle
    print(f"\n--- Starting Prediction from cycle {start_cycle} ---")
    with torch.no_grad():
        while True:
            first_input_cycle = current_cycle - cfg.LOOK_BACK_CYCLES
            if first_input_cycle < 0: break
            if (current_cycle * cfg.NUM_POINTS_PER_CYCLE) > all_timeseries_data.shape[0]: break

            input_features_list = []
            valid_lookback = True
            for cycle in range(first_input_cycle, current_cycle):
                start_idx = cycle * cfg.NUM_POINTS_PER_CYCLE
                end_idx = start_idx + cfg.NUM_POINTS_PER_CYCLE
                if end_idx > all_timeseries_data.shape[0]: valid_lookback = False; break
                cycle_features = all_timeseries_data[start_idx:end_idx, cfg.FEATURE_INDICES]
                if not np.all(np.isfinite(cycle_features)): valid_lookback = False; break
                input_features_list.append(torch.tensor(cycle_features, dtype=torch.float32).permute(1, 0))

            if not valid_lookback or not input_features_list: break

            stacked_features = torch.stack(input_features_list, dim=0).unsqueeze(0).to(cfg.DEVICE)
            predicted_value = model(stacked_features).item()
            predicted_soh[current_cycle] = predicted_value

            if predicted_value < cfg.EOL_THRESHOLD:
                print(f"Prediction stopped at cycle {current_cycle}: SOH reached EOL threshold.")
                break
            current_cycle += 1
    return predicted_soh

def calculate_mape(actual, predicted):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    finite_preds_indices = np.isfinite(predicted)
    actual = actual[finite_preds_indices]
    predicted = predicted[finite_preds_indices]
    non_zero_indices = np.abs(actual) > 1e-6
    if np.sum(non_zero_indices) == 0:
        return np.nan
    actual_final = actual[non_zero_indices]
    predicted_final = predicted[non_zero_indices]
    if actual_final.size == 0:
        return np.nan
    return np.mean(np.abs((actual_final - predicted_final) / actual_final)) * 100

def plot_single_battery_results(battery_id, true_soh_map, predicted_soh_map, prediction_start_cycle):
    """Plots and saves the SOH comparison for a single battery."""
    print(f"\n--- Generating Plot for Battery: {battery_id} ---")
    os.makedirs(cfg.FIG_SAVE_DIR, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    true_cycles, true_values = zip(*sorted(true_soh_map.items()))
    plt.plot(true_cycles, true_values, 'b-o', label='True SOH')
    if predicted_soh_map:
        pred_cycles, pred_values = zip(*sorted(predicted_soh_map.items()))
        plt.plot(pred_cycles, pred_values, 'r--x', label='Predicted SOH')
    plt.axhline(y=cfg.EOL_THRESHOLD, color='k', linestyle=':', label=f'EOL Threshold ({cfg.EOL_THRESHOLD})')
    plt.axvline(x=prediction_start_cycle - 0.5, color='g', linestyle='--', label=f'Prediction Start')
    plt.title(f"SOH Prediction vs True SOH for Battery: {battery_id}")
    plt.xlabel("Cycle Number")
    plt.ylabel("State of Health (SOH)")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=max(0, cfg.EOL_THRESHOLD - 0.1))
    save_path = os.path.join(cfg.FIG_SAVE_DIR, f"{battery_id}_soh_comparison.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

def plot_eol_comparison(actual_eols, predicted_eols, mape, save_path):
    """Plots the actual vs predicted EOL scatter plot."""
    print("\n--- Generating EOL Comparison Plot ---")
    actual_eols = np.array(actual_eols, dtype=float)
    predicted_eols = np.array(predicted_eols, dtype=float)
    valid_indices = np.isfinite(predicted_eols) & np.isfinite(actual_eols)
    if np.sum(valid_indices) == 0:
        print("Warning: No valid EOL pairs to plot.")
        return
    plt.figure(figsize=(8, 8), dpi=320)
    plt.scatter(actual_eols[valid_indices], predicted_eols[valid_indices], alpha=0.8, s=100, edgecolors='k')
    all_vals = np.concatenate((actual_eols[valid_indices], predicted_eols[valid_indices]))
    if all_vals.size == 0:
        plot_min, plot_max = 0, 100
    else:
        plot_min = np.min(all_vals) * 0.9
        plot_max = np.max(all_vals) * 1.1
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2.5, label='y=x')
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    plt.xlabel("Actual EOL (Cycles)")
    plt.ylabel("Predicted EOL (Cycles)")
    title = f"EOL Prediction vs Actual ({cfg.MODEL_TYPE})\nOverall MAPE: {mape:.2f}%" if not np.isnan(mape) else f"EOL Prediction vs Actual ({cfg.MODEL_TYPE})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Scatter plot saved to {save_path}")
    plt.close()