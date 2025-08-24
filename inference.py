#!/usr/bin/env python3
"""
电池SOH推理脚本

该脚本使用训练好的模型对输入的NPZ文件进行SOH预测，
输出每个循环周期的预测SOH值。

使用方法:
    python inference.py --input_file path/to/input.npz --model_path path/to/model.pth --output_file results.csv

功能:
- 自动检测输入NPZ文件格式（原始格式或增强格式）
- 如果是原始格式，自动转换为增强格式
- 使用训练好的模型进行SOH预测
- 输出预测结果到CSV文件
"""

import os
import argparse
import numpy as np
import torch
import csv
import tempfile
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List

# 导入项目模块
import src.config as cfg
from src.model import CNNLSTM, CNNTransformer
from npz_converter import convert_npz_file


def detect_npz_format(file_path: str) -> str:
    """
    检测NPZ文件格式
    
    Args:
        file_path: NPZ文件路径
        
    Returns:
        'enhanced' 如果是增强格式, 'original' 如果是原始格式
        
    Raises:
        ValueError: 如果文件格式无法识别
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        keys = list(data.keys())
        
        # 检查是否是增强格式
        if 'timeseries_data_enhanced' in keys and 'soh_per_cycle' in keys and 'eol' in keys:
            return 'enhanced'
        
        # 检查是否是原始格式
        elif 'discharge_timeseries' in keys and 'cycle_summary' in keys:
            return 'original'
        
        else:
            raise ValueError(f"无法识别的NPZ文件格式，包含键: {keys}")
            
    except Exception as e:
        raise ValueError(f"读取NPZ文件失败: {e}")


def convert_to_enhanced_format(input_file: str) -> str:
    """
    将原始NPZ格式转换为增强格式
    
    Args:
        input_file: 原始NPZ文件路径
        
    Returns:
        临时增强NPZ文件路径
        
    Raises:
        RuntimeError: 如果转换失败
    """
    print(f"检测到原始格式，正在转换为增强格式...")
    
    # 创建临时文件
    temp_fd, temp_path = tempfile.mkstemp(suffix='_enhanced.npz')
    os.close(temp_fd)  # 关闭文件描述符，只保留路径
    
    try:
        success = convert_npz_file(input_file, temp_path)
        if not success:
            raise RuntimeError("NPZ格式转换失败")
        
        print(f"格式转换成功，临时文件: {temp_path}")
        return temp_path
        
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"格式转换失败: {e}")


def load_enhanced_data(file_path: str) -> Tuple[np.ndarray, Dict[int, float], float, int]:
    """
    加载增强格式的NPZ数据
    
    Args:
        file_path: 增强NPZ文件路径
        
    Returns:
        (timeseries_data, soh_map, eol, num_cycles)
    """
    try:
        data = np.load(file_path)
        timeseries_data = data['timeseries_data_enhanced']
        soh_per_cycle = data['soh_per_cycle']
        eol = float(data['eol'])
        
        # 创建SOH映射
        soh_map = {int(cycle): float(soh) for cycle, soh in soh_per_cycle}
        
        # 计算总周期数
        if len(timeseries_data) > 0:
            max_cycle = int(timeseries_data[:, 0].max())
            num_cycles = max_cycle + 1
        else:
            num_cycles = len(soh_map)
        
        return timeseries_data, soh_map, eol, num_cycles
        
    except Exception as e:
        raise RuntimeError(f"加载增强数据失败: {e}")


def load_model(model_path: str) -> torch.nn.Module:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        加载的模型
    """
    print(f"正在加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    try:
        # 创建模型实例
        if cfg.MODEL_TYPE == 'CNNLSTM':
            model = CNNLSTM(
                input_features=cfg.INPUT_FEATURES,
                cnn_out_channels=cfg.CNN_OUT_CHANNELS,
                kernel_size=cfg.CNN_KERNEL_SIZE,
                lstm_hidden_size=cfg.LSTM_HIDDEN_SIZE,
                lstm_num_layers=cfg.LSTM_NUM_LAYERS,
                sequence_length=cfg.NUM_POINTS_PER_CYCLE
            )
        elif cfg.MODEL_TYPE == 'CNNTransformer':
            model = CNNTransformer(
                input_features=cfg.INPUT_FEATURES,
                cnn_out_channels=cfg.CNN_OUT_CHANNELS,
                kernel_size=cfg.CNN_KERNEL_SIZE,
                transformer_nhead=cfg.TRANSFORMER_NHEAD,
                transformer_num_encoder_layers=cfg.TRANSFORMER_NUM_ENCODER_LAYERS,
                transformer_dim_feedforward=cfg.TRANSFORMER_DIM_FEEDFORWARD,
                sequence_length=cfg.NUM_POINTS_PER_CYCLE,
                dropout=cfg.TRANSFORMER_DROPOUT
            )
        else:
            raise ValueError(f"未知的模型类型: {cfg.MODEL_TYPE}")
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=cfg.DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(cfg.DEVICE)
        model.eval()
        
        print(f"模型加载成功，类型: {cfg.MODEL_TYPE}")
        return model
        
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")


def predict_all_cycles_soh(
    model: torch.nn.Module, 
    timeseries_data: np.ndarray, 
    start_cycle: int = None
) -> Dict[int, float]:
    """
    预测所有可能周期的SOH
    
    Args:
        model: 训练好的模型
        timeseries_data: 增强时间序列数据
        start_cycle: 开始预测的周期（如果为None，从look_back_cycles开始）
        
    Returns:
        预测的SOH字典 {cycle: soh}
    """
    print("开始SOH预测...")
    
    # 确定开始周期
    if start_cycle is None:
        start_cycle = cfg.LOOK_BACK_CYCLES
    
    # 计算总周期数
    max_cycle = int(timeseries_data[:, 0].max())
    total_cycles = max_cycle + 1
    
    predicted_soh = {}
    
    print(f"预测范围: 周期 {start_cycle} 到 {total_cycles-1}")
    
    with torch.no_grad():
        for current_cycle in range(start_cycle, total_cycles):
            # 检查是否有足够的历史数据
            first_input_cycle = current_cycle - cfg.LOOK_BACK_CYCLES
            if first_input_cycle < 0:
                continue
            
            # 检查数据是否足够
            required_end_idx = current_cycle * cfg.NUM_POINTS_PER_CYCLE
            if required_end_idx > len(timeseries_data):
                break
            
            # 准备输入特征
            input_features_list = []
            valid_data = True
            
            for cycle in range(first_input_cycle, current_cycle):
                start_idx = cycle * cfg.NUM_POINTS_PER_CYCLE
                end_idx = start_idx + cfg.NUM_POINTS_PER_CYCLE
                
                if end_idx > len(timeseries_data):
                    valid_data = False
                    break
                
                # 提取特征（电压、容量、dQ/dV）
                cycle_features = timeseries_data[start_idx:end_idx, cfg.FEATURE_INDICES]
                
                # 检查数据有效性
                if not np.all(np.isfinite(cycle_features)):
                    valid_data = False
                    break
                
                # 转换为张量并调整维度
                features_tensor = torch.tensor(cycle_features, dtype=torch.float32).permute(1, 0)
                input_features_list.append(features_tensor)
            
            if not valid_data or not input_features_list:
                print(f"跳过周期 {current_cycle}: 数据无效")
                continue
            
            try:
                # 堆叠特征并添加批次维度
                stacked_features = torch.stack(input_features_list, dim=0).unsqueeze(0).to(cfg.DEVICE)
                
                # 进行预测
                prediction = model(stacked_features)
                predicted_value = prediction.item()
                
                predicted_soh[current_cycle] = predicted_value
                
                # 如果预测值低于EOL阈值，可以选择停止预测
                if predicted_value < cfg.EOL_THRESHOLD:
                    print(f"周期 {current_cycle}: SOH达到EOL阈值 ({predicted_value:.4f} < {cfg.EOL_THRESHOLD})")
                
            except Exception as e:
                print(f"预测周期 {current_cycle} 时出错: {e}")
                continue
    
    print(f"预测完成，成功预测 {len(predicted_soh)} 个周期")
    return predicted_soh


def save_predictions_to_csv(
    predictions: Dict[int, float], 
    output_file: str, 
    true_soh: Dict[int, float] = None,
    battery_id: str = "unknown"
) -> None:
    """
    保存预测结果到CSV文件
    
    Args:
        predictions: 预测的SOH字典
        output_file: 输出CSV文件路径
        true_soh: 真实SOH字典（可选）
        battery_id: 电池ID
    """
    print(f"保存预测结果到: {output_file}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 准备数据
    all_cycles = sorted(set(predictions.keys()) | (set(true_soh.keys()) if true_soh else set()))
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        if true_soh:
            writer.writerow(['Battery_ID', 'Cycle', 'True_SOH', 'Predicted_SOH', 'Absolute_Error'])
        else:
            writer.writerow(['Battery_ID', 'Cycle', 'Predicted_SOH'])
        
        # 写入数据
        for cycle in all_cycles:
            pred_soh = predictions.get(cycle, None)
            true_val = true_soh.get(cycle, None) if true_soh else None
            
            if true_soh and true_val is not None and pred_soh is not None:
                abs_error = abs(true_val - pred_soh)
                writer.writerow([battery_id, cycle, f"{true_val:.6f}", f"{pred_soh:.6f}", f"{abs_error:.6f}"])
            elif pred_soh is not None:
                if true_soh:
                    writer.writerow([battery_id, cycle, "", f"{pred_soh:.6f}", ""])
                else:
                    writer.writerow([battery_id, cycle, f"{pred_soh:.6f}"])
            elif true_val is not None and true_soh:
                writer.writerow([battery_id, cycle, f"{true_val:.6f}", "", ""])
    
    print(f"结果已保存，共 {len(all_cycles)} 行数据")


def plot_predictions(
    predictions: Dict[int, float], 
    true_soh: Dict[int, float] = None,
    output_path: str = None,
    battery_id: str = "unknown"
) -> None:
    """
    绘制预测结果图表
    
    Args:
        predictions: 预测SOH字典
        true_soh: 真实SOH字典（可选）
        output_path: 图片保存路径
        battery_id: 电池ID
    """
    print("正在生成预测结果图表...")
    
    plt.figure(figsize=(12, 6))
    
    # 绘制预测值
    if predictions:
        pred_cycles, pred_values = zip(*sorted(predictions.items()))
        plt.plot(pred_cycles, pred_values, 'r--o', label='预测SOH', markersize=4)
    
    # 绘制真实值
    if true_soh:
        true_cycles, true_values = zip(*sorted(true_soh.items()))
        plt.plot(true_cycles, true_values, 'b-s', label='真实SOH', markersize=3)
    
    # 添加EOL阈值线
    plt.axhline(y=cfg.EOL_THRESHOLD, color='k', linestyle=':', 
                label=f'EOL阈值 ({cfg.EOL_THRESHOLD})', alpha=0.7)
    
    plt.title(f'电池 {battery_id} SOH预测结果', fontsize=14)
    plt.xlabel('循环周期', fontsize=12)
    plt.ylabel('健康状态 (SOH)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 设置Y轴范围
    all_values = list(predictions.values())
    if true_soh:
        all_values.extend(true_soh.values())
    
    if all_values:
        y_min = max(0, min(all_values) - 0.05)
        y_max = min(1.1, max(all_values) + 0.05)
        plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='电池SOH推理脚本')
    parser.add_argument('--input_file', type=str, required=True, 
                       help='输入NPZ文件路径')
    parser.add_argument('--model_path', type=str, 
                       default=os.path.join(cfg.MODEL_SAVE_DIR, cfg.PRE_TRAINED_MODEL_NAME),
                       help='训练好的模型文件路径')
    parser.add_argument('--output_file', type=str, default='predictions.csv',
                       help='输出CSV文件路径')
    parser.add_argument('--plot_output', type=str, default=None,
                       help='输出图表文件路径（可选）')
    parser.add_argument('--start_cycle', type=int, default=None,
                       help='开始预测的周期（默认为look_back_cycles）')
    
    args = parser.parse_args()
    
    print("=== 电池SOH推理开始 ===")
    print(f"输入文件: {args.input_file}")
    print(f"模型路径: {args.model_path}")
    print(f"输出文件: {args.output_file}")
    
    temp_file = None
    
    try:
        # 1. 检测文件格式
        file_format = detect_npz_format(args.input_file)
        print(f"检测到文件格式: {file_format}")
        
        # 2. 如果是原始格式，转换为增强格式
        if file_format == 'original':
            enhanced_file = convert_to_enhanced_format(args.input_file)
            temp_file = enhanced_file  # 记录临时文件以便清理
        else:
            enhanced_file = args.input_file
        
        # 3. 加载增强数据
        timeseries_data, true_soh_map, eol, num_cycles = load_enhanced_data(enhanced_file)
        print(f"数据加载成功: {num_cycles} 个周期, EOL={eol}")
        
        # 4. 加载模型
        model = load_model(args.model_path)
        
        # 5. 进行SOH预测
        predictions = predict_all_cycles_soh(model, timeseries_data, args.start_cycle)
        
        if not predictions:
            print("警告: 没有生成任何有效的预测结果")
            return
        
        # 6. 保存结果
        battery_id = os.path.splitext(os.path.basename(args.input_file))[0]
        save_predictions_to_csv(predictions, args.output_file, true_soh_map, battery_id)
        
        # 7. 生成图表
        if args.plot_output:
            plot_predictions(predictions, true_soh_map, args.plot_output, battery_id)
        
        # 8. 输出统计信息
        print(f"\n=== 预测统计 ===")
        print(f"预测周期范围: {min(predictions.keys())} - {max(predictions.keys())}")
        print(f"预测SOH范围: {min(predictions.values()):.4f} - {max(predictions.values()):.4f}")
        
        # 计算误差（如果有真实值）
        if true_soh_map:
            common_cycles = set(predictions.keys()) & set(true_soh_map.keys())
            if common_cycles:
                errors = [abs(predictions[c] - true_soh_map[c]) for c in common_cycles]
                mae = np.mean(errors)
                rmse = np.sqrt(np.mean([e**2 for e in errors]))
                print(f"与真实值比较 ({len(common_cycles)} 个周期):")
                print(f"  平均绝对误差 (MAE): {mae:.6f}")
                print(f"  均方根误差 (RMSE): {rmse:.6f}")
        
        print("\n=== 推理完成 ===")
        
    except Exception as e:
        print(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"临时文件已清理: {temp_file}")
            except:
                print(f"警告: 无法删除临时文件: {temp_file}")


if __name__ == '__main__':
    main()
