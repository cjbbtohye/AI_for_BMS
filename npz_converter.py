import numpy as np
import os
import glob
import argparse
import time
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict


def calculate_dqdv(voltage, capacity, clip_value=100):
    """
    计算差分容量 dQ/dV
    
    参数:
        voltage: 电压数组
        capacity: 容量数组
        clip_value: 裁剪值，防止数值不稳定
    
    返回:
        dQ/dV 数组
    """
    dV = np.diff(voltage)
    dQ = np.diff(capacity)
    
    # 处理电压差过小的情况
    dqdv = np.zeros_like(dV)
    valid_mask = np.abs(dV) >= 1e-6
    dqdv[valid_mask] = dQ[valid_mask] / dV[valid_mask]
    
    # 裁剪极值
    dqdv = np.clip(dqdv, -clip_value, clip_value)
    
    # 在开头添加一个值以保持长度一致
    dqdv = np.concatenate([[dqdv[0] if len(dqdv) > 0 else 0.0], dqdv])
    
    return dqdv


def interpolate_cycle_data(cycle_data, target_points=1000):
    """
    将单个周期的数据插值到固定点数
    
    参数:
        cycle_data: 单个周期的数据 (n_points, n_features)
        target_points: 目标点数
    
    返回:
        插值后的数据 (target_points, n_features)
    """
    if len(cycle_data) < 2:
        return None
    
    # 提取原始数据
    original_cycle_fraction = cycle_data[:, 1]  # intra_process_cycle_fraction
    voltage = cycle_data[:, 2]  # voltage_v
    capacity = cycle_data[:, 4]  # capacity_ah
    
    # 确保数据按cycle_fraction排序
    sort_idx = np.argsort(original_cycle_fraction)
    original_cycle_fraction = original_cycle_fraction[sort_idx]
    voltage = voltage[sort_idx]
    capacity = capacity[sort_idx]
    
    # 规范化cycle_fraction到0-1范围
    fraction_min = original_cycle_fraction.min()
    fraction_max = original_cycle_fraction.max()
    
    if fraction_max - fraction_min <= 0:
        print(f"周期数据范围无效: {fraction_min} - {fraction_max}")
        return None
    
    normalized_fraction = (original_cycle_fraction - fraction_min) / (fraction_max - fraction_min)
    
    # 创建目标网格
    target_fraction = np.linspace(0, 1, target_points)
    
    try:
        # 插值电压和容量
        voltage_interp = interp1d(normalized_fraction, voltage, 
                                kind='linear', bounds_error=False, fill_value='extrapolate')
        capacity_interp = interp1d(normalized_fraction, capacity, 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        
        voltage_interpolated = voltage_interp(target_fraction)
        capacity_interpolated = capacity_interp(target_fraction)
        
        # 检查插值结果的合理性
        if np.any(np.isnan(voltage_interpolated)) or np.any(np.isnan(capacity_interpolated)):
            print("插值结果包含NaN值")
            return None
        
        # 应用高斯平滑
        voltage_smoothed = gaussian_filter1d(voltage_interpolated, sigma=1.0)
        capacity_smoothed = gaussian_filter1d(capacity_interpolated, sigma=1.0)
        
        # 验证结果范围的合理性
        if voltage_smoothed.max() > 10 or voltage_smoothed.min() < 0:
            print(f"电压范围异常: {voltage_smoothed.min():.3f} - {voltage_smoothed.max():.3f}")
            return None
        
        return voltage_smoothed, capacity_smoothed, target_fraction
        
    except Exception as e:
        print(f"插值失败: {e}")
        return None


def get_soh_for_cycle(cycle_num, soh_data):
    """
    获取指定周期的SOH值
    """
    if len(soh_data) == 0:
        return 0.0
    
    # 找到对应周期的SOH
    cycle_numbers = soh_data[:, 0]
    soh_values = soh_data[:, 1]
    
    # 如果找到精确匹配
    exact_match = np.where(cycle_numbers == cycle_num)[0]
    if len(exact_match) > 0:
        return soh_values[exact_match[0]]
    
    # 如果没有精确匹配，使用最近的值
    if cycle_num < cycle_numbers.min():
        return soh_values[0]
    elif cycle_num > cycle_numbers.max():
        return soh_values[-1]
    else:
        # 插值
        return np.interp(cycle_num, cycle_numbers, soh_values)


def convert_npz_file(input_path, output_path):
    """
    转换单个NPZ文件到目标格式
    """
    try:
        # 加载原始数据
        data = np.load(input_path, allow_pickle=True)
        
        # 检查必要的键
        required_keys = ['discharge_timeseries', 'cycle_summary', 'eol_info']
        for key in required_keys:
            if key not in data:
                print(f"跳过 {input_path}: 缺少键 '{key}'")
                return False
        
        discharge_ts = data['discharge_timeseries']
        cycle_summary = data['cycle_summary']
        eol_info = data['eol_info'].item()
        
        # 检查EOL是否有效
        eol_value = eol_info.get('eol_cycle_number', np.nan)
        if np.isnan(eol_value):
            print(f"跳过 {input_path}: EOL值为NaN")
            return False
        
        # 检查时间序列数据是否为空
        if len(discharge_ts) == 0:
            print(f"跳过 {input_path}: 时间序列数据为空")
            return False
        
        # 提取SOH数据
        soh_per_cycle = cycle_summary[:, [0, 1]]  # cycle_number, soh
        
        # 按周期分组数据
        cycle_groups = defaultdict(list)
        for row in discharge_ts:
            cycle_num = int(row[0])  # parent_cycle_number
            cycle_groups[cycle_num].append(row)
        
        # 转换为numpy数组
        for cycle_num in cycle_groups:
            cycle_groups[cycle_num] = np.array(cycle_groups[cycle_num])
        
        # 处理每个周期
        enhanced_data_list = []
        
        for cycle_num in sorted(cycle_groups.keys()):
            cycle_data = cycle_groups[cycle_num]
            
            # 跳过数据点太少的周期
            if len(cycle_data) < 2:
                print(f"跳过周期 {cycle_num}: 数据点不足")
                continue
            
            # 插值数据
            interpolation_result = interpolate_cycle_data(cycle_data, target_points=1000)
            if interpolation_result is None:
                print(f"跳过周期 {cycle_num}: 插值失败")
                continue
            
            voltage_smoothed, capacity_smoothed, target_fraction = interpolation_result
            
            # 创建周期数组
            cycle_array = np.full(1000, cycle_num, dtype=np.float32)
            
            # 创建discharge_cycle数组 (从cycle_num到cycle_num+1)
            discharge_cycle = np.linspace(cycle_num, cycle_num + 1, 1000, dtype=np.float32)
            
            # 获取当前和下一个周期的SOH
            current_soh = get_soh_for_cycle(cycle_num, soh_per_cycle)
            next_soh = get_soh_for_cycle(cycle_num + 1, soh_per_cycle)
            
            # 插值SOH
            soh_interpolated = current_soh + (next_soh - current_soh) * target_fraction
            soh_interpolated = np.nan_to_num(soh_interpolated, nan=0.0).astype(np.float32)
            
            # 计算dQ/dV
            dqdv = calculate_dqdv(voltage_smoothed, capacity_smoothed)
            dqdv = np.nan_to_num(dqdv, nan=0.0).astype(np.float32)
            
            # 组合数据
            cycle_enhanced = np.column_stack([
                cycle_array,                    # Column 0: Cycle Number
                discharge_cycle,                # Column 1: Discharge Cycle
                voltage_smoothed.astype(np.float32),  # Column 2: Voltage
                capacity_smoothed.astype(np.float32), # Column 3: Capacity
                soh_interpolated,               # Column 4: Interpolated SOH
                dqdv                           # Column 5: dQ/dV
            ])
            
            enhanced_data_list.append(cycle_enhanced)
        
        if not enhanced_data_list:
            print(f"跳过 {input_path}: 没有有效的周期数据")
            return False
        
        # 合并所有周期的数据
        timeseries_data_enhanced = np.vstack(enhanced_data_list)
        
        # 准备输出数据
        output_data = {
            'timeseries_data_enhanced': timeseries_data_enhanced,
            'soh_per_cycle': soh_per_cycle.astype(np.float32),
            'eol': np.float32(eol_value)
        }
        
        # 保存到输出文件
        np.savez_compressed(output_path, **output_data)
        
        print(f"成功转换: {input_path} -> {output_path}")
        print(f"  输出形状: {timeseries_data_enhanced.shape}")
        print(f"  周期数: {len(enhanced_data_list)}")
        
        return True
        
    except Exception as e:
        print(f"转换失败 {input_path}: {e}")
        # 清理部分输出文件
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def main():
    parser = argparse.ArgumentParser(description='转换NPZ文件到增强格式')
    parser.add_argument('--data_dir', required=True, help='输入NPZ文件目录')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有NPZ文件
    input_files = glob.glob(os.path.join(args.data_dir, '*.npz'))
    
    if not input_files:
        print(f"在 {args.data_dir} 中没有找到NPZ文件")
        return
    
    print(f"找到 {len(input_files)} 个NPZ文件")
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, input_file in enumerate(input_files, 1):
        print(f"\n处理 {i}/{len(input_files)}: {os.path.basename(input_file)}")
        
        # 创建输出文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(args.output_dir, f"{base_name}_enhanced.npz")
        
        # 转换文件
        if convert_npz_file(input_file, output_file):
            successful += 1
        else:
            failed += 1
    
    # 输出统计信息
    end_time = time.time()
    print(f"\n=== 转换完成 ===")
    print(f"总文件数: {len(input_files)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    print(f"总用时: {end_time - start_time:.2f} 秒")


if __name__ == '__main__':
    main()
