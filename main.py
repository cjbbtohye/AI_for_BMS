from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging
import json
import os
import base64
import traceback
from io import BytesIO
from pathlib import Path
import math
import graphviz
import numpy as np
import torch
import tempfile
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import matplotlib.font_manager as fm
# 导入项目模块
import src.config as cfg
from src.model import CNNLSTM, CNNTransformer
from npz_converter import convert_npz_file

from typing import List

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建MCP服务器对象
mcp = FastMCP()

def read_battery_data(folder_name: str):
    """读取指定文件夹中的电池包和BMS拓扑数据"""
    batteries_dir = Path(folder_name)
    
    if not batteries_dir.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_name}")
    
    # 读取电池包信息
    battery_pack_file = batteries_dir / "battery_pack.json"
    bms_topology_file = batteries_dir / "bms_topology.json"
    
    if not battery_pack_file.exists():
        raise FileNotFoundError(f"在文件夹 {folder_name} 中未找到 battery_pack.json")
    
    if not bms_topology_file.exists():
        raise FileNotFoundError(f"在文件夹 {folder_name} 中未找到 bms_topology.json")
    
    with open(battery_pack_file, 'r', encoding='utf-8') as f:
        battery_data = json.load(f)
    
    with open(bms_topology_file, 'r', encoding='utf-8') as f:
        topology_data = json.load(f)
    
    return battery_data, topology_data

def create_bms_topology_graphviz(topology_data, battery_data):
    """使用Graphviz创建BMS拓扑的可视化图表"""
    dot = graphviz.Digraph('BMS_Topology', comment='BMS Topology')
    dot.attr(rankdir='TB', splines='ortho', overlap='false', nodesep='1.0', ranksep='1.5', 
             ratio='auto', size='14,10', bgcolor='white', concentrate='true')

    pack_info = battery_data["packInfo"]
    config = battery_data['configuration']
    specs = battery_data['packSpecifications']
    
    title = f'📋 {pack_info["packName"]}\\n' \
            f'⚡ 配置: {config["seriesCount"]}S{config["parallelCount"]}P | ' \
            f'🔋 电压: {specs["nominalVoltage_V"]}V | ' \
            f'🔌 容量: {specs["nominalCapacity_Ah"]}Ah\\n' \
            f'🏭 制造商: {pack_info.get("manufacturer", "N/A")}'
    dot.attr('graph', label=title, labelloc='t', fontsize='18', fontname='Arial Unicode MS', 
             margin='0.5', pad='0.5')

    # 兼容不同的数据结构格式
    bms_units = topology_data.get('batterySystem', {}).get('bmsTopology', {}).get('bmsUnits', topology_data.get('bmsUnits', []))
    
    # 按角色分类节点，支持更多角色类型
    role_mapping = {
        'master': {'name': '🎛️ 主控', 'color': '#87CEEB', 'shape': 'box', 'style': 'filled,bold'},
        'slave': {'name': '📡 从控', 'color': '#FFE4B5', 'shape': 'ellipse', 'style': 'filled'},
        'central_controller': {'name': '🏢 中央控制器', 'color': '#98FB98', 'shape': 'box', 'style': 'filled,bold'},
        'node': {'name': '📊 监控节点', 'color': '#F0A0A0', 'shape': 'ellipse', 'style': 'filled'},
        'controller': {'name': '🎮 控制器', 'color': '#87CEEB', 'shape': 'box', 'style': 'filled,bold'},
        'monitor': {'name': '👁️ 监控器', 'color': '#FFE4B5', 'shape': 'ellipse', 'style': 'filled'}
    }
    
    categorized_units = {}
    for unit in bms_units:
        role = unit.get('role', 'unknown')
        if role not in categorized_units:
            categorized_units[role] = []
        categorized_units[role].append(unit)

    # 创建不同角色的节点
    for role, units in categorized_units.items():
        role_info = role_mapping.get(role, {'name': role, 'color': 'seashell', 'shape': 'ellipse'})
        
        # 对于控制器类型（master, central_controller, controller），不创建集群
        if role in ['master', 'central_controller', 'controller']:
            for unit in units:
                cell_count = len(unit.get('monitoredCellIds', []))
                label = f"{role_info['name']}: {unit['unitId']}\\n" \
                        f"📦 型号: {unit.get('model', 'N/A')}\\n" \
                        f"🔢 监控电芯: {cell_count}"
                
                # 为central_controller添加更多信息
                if role == 'central_controller':
                    features = unit.get('features', {})
                    if features:
                        feature_list = []
                        if features.get('cellBalancing'): feature_list.append('⚖️均衡')
                        if features.get('temperatureMonitoring'): feature_list.append('🌡️温度')
                        if features.get('insulationMonitoring'): feature_list.append('🛡️绝缘')
                        if features.get('dataLogging'): feature_list.append('📊数据记录')
                        if feature_list:
                            label += f"\\n🔧 功能: {', '.join(feature_list)}"
                
                dot.node(unit['unitId'], label, shape=role_info['shape'], 
                        style=role_info.get('style', 'filled'), fillcolor=role_info['color'], 
                        fontname='Arial Unicode MS', penwidth='2')
        
        # 对于其他类型创建集群
        elif len(units) > 1:
            with dot.subgraph(name=f'cluster_{role}') as c:
                c.attr(label=f'{role_info["name"]}单元', style='filled', color='#F5F5F5', 
                       fontname='Arial Unicode MS', penwidth='2', bgcolor='#FAFAFA')
                for unit in units:
                    cell_count = len(unit.get('monitoredCellIds', []))
                    label = f"{role_info['name']}: {unit['unitId']}\\n" \
                            f"📦 型号: {unit.get('model', 'N/A')}\\n" \
                            f"🔢 监控电芯: {cell_count}"
                    c.node(unit['unitId'], label, shape=role_info['shape'], 
                          style=role_info.get('style', 'filled'), fillcolor=role_info['color'], 
                          fontname='Arial Unicode MS', penwidth='2')
        else:
            # 单个节点直接创建
            for unit in units:
                cell_count = len(unit.get('monitoredCellIds', []))
                label = f"{role_info['name']}: {unit['unitId']}\\n" \
                        f"🏷️ 角色: {role}\\n" \
                        f"📦 型号: {unit.get('model', 'N/A')}\\n" \
                        f"🔢 监控电芯: {cell_count}"
                dot.node(unit['unitId'], label, shape=role_info['shape'], 
                        style=role_info.get('style', 'filled'), fillcolor=role_info['color'], 
                        fontname='Arial Unicode MS', penwidth='2')

    # 创建连接
    for unit in bms_units:
        unit_id = unit['unitId']
        connections = unit.get('connections', {})
        
        # 主控 -> 从控连接
        if 'controlsUnits' in connections:
            for controlled_unit_id in connections['controlsUnits']:
                dot.edge(unit_id, controlled_unit_id, label='🔗 CAN Bus', 
                        fontname='Arial Unicode MS', color='#4169E1', penwidth='2')

        # 菊花链连接
        if 'daisyChainNext' in connections and connections['daisyChainNext']:
            protocol = connections.get('protocol', 'Chain')
            dot.edge(unit_id, connections['daisyChainNext'], 
                    label=f'⛓️ {protocol}', style='dashed', fontname='Arial Unicode MS', 
                    color='#FF6347', penwidth='2')
        
        # 处理外部接口（针对central_controller）
        external_interfaces = connections.get('externalInterfaces', [])
        if external_interfaces:
            # 为外部接口创建一个虚拟节点
            ext_node_id = f"{unit_id}_external"
            ext_label = "🔌 外部接口\\n"
            for interface in external_interfaces:
                ext_label += f"📡 {interface.get('type', 'N/A')}: {interface.get('purpose', 'N/A')}\\n"
            
            dot.node(ext_node_id, ext_label.strip(), shape='note', 
                    style='filled', fillcolor='#F0F8FF', fontname='Arial Unicode MS',
                    penwidth='2', color='#4682B4')
            dot.edge(unit_id, ext_node_id, label='🔌 接口', style='dotted', 
                    fontname='Arial Unicode MS', color='#32CD32', penwidth='2')

    return dot

def detect_npz_format(file_path: str) -> str:
    """检测NPZ文件格式"""
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
    """将原始NPZ格式转换为增强格式"""
    logger.info(f"检测到原始格式，正在转换为增强格式...")
    
    # 创建临时文件
    temp_fd, temp_path = tempfile.mkstemp(suffix='_enhanced.npz')
    os.close(temp_fd)  # 关闭文件描述符，只保留路径
    
    try:
        success = convert_npz_file(input_file, temp_path)
        if not success:
            raise RuntimeError("NPZ格式转换失败")
        
        logger.info(f"格式转换成功，临时文件: {temp_path}")
        return temp_path
        
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"格式转换失败: {e}")

def load_enhanced_data(file_path: str) -> Tuple[np.ndarray, Dict[int, float], float, int]:
    """加载增强格式的NPZ数据"""
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

def load_model_for_inference(model_path: str) -> torch.nn.Module:
    """加载训练好的模型"""
    logger.info(f"正在加载模型: {model_path}")
    
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
        
        logger.info(f"模型加载成功，类型: {cfg.MODEL_TYPE}")
        return model
        
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")

def predict_soh_for_visualization(
    model: torch.nn.Module, 
    timeseries_data: np.ndarray, 
    start_cycle: int = None
) -> Dict[int, float]:
    """预测所有可能周期的SOH"""
    logger.info("开始SOH预测...")
    
    # 确定开始周期
    if start_cycle is None:
        start_cycle = cfg.LOOK_BACK_CYCLES
    
    # 计算总周期数
    max_cycle = int(timeseries_data[:, 0].max())
    total_cycles = max_cycle + 1
    
    predicted_soh = {}
    
    logger.info(f"预测范围: 周期 {start_cycle} 到 {total_cycles-1}")
    
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
                continue
            
            try:
                # 堆叠特征并添加批次维度
                stacked_features = torch.stack(input_features_list, dim=0).unsqueeze(0).to(cfg.DEVICE)
                
                # 进行预测
                prediction = model(stacked_features)
                predicted_value = prediction.item()
                
                predicted_soh[current_cycle] = predicted_value
                
            except Exception as e:
                logger.warning(f"预测周期 {current_cycle} 时出错: {e}")
                continue
    
    logger.info(f"预测完成，成功预测 {len(predicted_soh)} 个周期")
    return predicted_soh

def create_soh_plot(predictions: Dict[int, float], battery_id: str) -> str:
    """创建SOH预测图表并返回base64编码的图片"""
    logger.info("正在生成SOH预测结果图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制预测值
    if predictions:
        pred_cycles, pred_values = zip(*sorted(predictions.items()))
        ax.plot(pred_cycles, pred_values, 'r--o', label='Predicted SOH', markersize=4, linewidth=2)
    
    # 添加EOL阈值线
    ax.axhline(y=cfg.EOL_THRESHOLD, color='k', linestyle=':', 
                label=f'EOL Threshold ({cfg.EOL_THRESHOLD})', alpha=0.7, linewidth=2)
    
    ax.set_title(f'Battery {battery_id} SOH Prediction Results', fontsize=16, fontweight='bold')
    ax.set_xlabel('Cycle', fontsize=14)
    ax.set_ylabel('State of Health (SOH)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 设置Y轴范围
    if predictions:
        all_values = list(predictions.values())
        y_min = max(0, min(all_values) - 0.05)
        y_max = min(1.1, max(all_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    # 添加预测统计信息
    if predictions:
        min_soh = min(predictions.values())
        max_soh = max(predictions.values())
        cycles_range = f"{min(predictions.keys())}-{max(predictions.keys())}"
        
        info_text = f'预测周期: {cycles_range}\nSOH范围: {min_soh:.3f}-{max_soh:.3f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存到内存中的字节流
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    # 转换为base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    plt.close(fig)
    buffer.close()
    
    logger.info("SOH预测图表生成完成")
    return image_base64

@mcp.tool()
def visualize_battery_system(folder_name: str) -> ImageContent:
    """生成指定电池包文件夹的可视化图表
    
    Args:
        folder_name: 文件夹的绝对路径，包含battery_pack.json和bms_topology.json文件
    
    Returns:
        包含电池包配置和BMS拓扑图的PNG图片（base64编码）
    """
    try:
        logger.info(f'开始生成电池系统可视化: {folder_name}')
        
        # 读取电池数据
        battery_data, topology_data = read_battery_data(folder_name)
        logger.info(f'数据读取成功: {folder_name}')
        
        # 创建可视化图表
        logger.info(f'开始创建Graphviz可视化图表: {folder_name}')
        dot = create_bms_topology_graphviz(topology_data, battery_data)
        logger.info(f'Graphviz图表对象创建成功: {folder_name}')
        
        # 在内存中渲染图像并进行base64编码
        image_data = dot.pipe(format='png')
        logger.info(f'Graphviz图表已在内存中渲染')
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        logger.info(f'base64转换成功，数据长度: {len(image_base64)}')
        
        logger.info(f'成功生成电池系统可视化图表: {folder_name}')
        return ImageContent(data=image_base64, mimeType="image/png", type="image")
        
    except Exception as e:
        logger.exception(f'生成可视化图表时出错: {str(e)}')
        # 创建一个简单的错误信息文本
        error_message = f'生成可视化图表时出错:\n\n{traceback.format_exc()}'
        # 这里可以考虑生成一个错误的图片返回，但为简化，我们直接返回文本错误
        # 为了符合返回类型，我们还是生成一个错误图片
        try:
            import graphviz
            dot = graphviz.Digraph('Error')
            dot.node('Error', error_message, shape='box', style='filled', fillcolor='red')
            error_image_data = dot.pipe(format='png')
            error_base64 = base64.b64encode(error_image_data).decode('utf-8')
            return ImageContent(data=error_base64, mimeType="image/png", type="image")
        except Exception as final_e:
             logger.error(f'创建错误图片时也失败了: {str(final_e)}')
             # 如果连graphviz都失败了，返回一个纯文本错误（虽然类型不匹配，但能传递信息）
             # 在实际应用中，这里应该有一个更优雅的降级方案
             return ImageContent(data="", mimeType="image/png", type="image")


@mcp.tool()
def list_available_battery_folders(file_path: str ) -> TextContent:
    """列出Batteries目录下所有可用的电池包文件夹
    Args:
        file_path: Batteries目录
    Returns:
        可用文件夹列表的文本描述
    """
    try:
        batteries_dir = Path(file_path)  #Path(__file__).parent / "Batteries"
        
        if not batteries_dir.exists():
            return TextContent(type="text", text="Batteries目录不存在")
        
        folders = []
        for item in batteries_dir.iterdir():
            if item.is_dir():
                # 检查是否包含必需的文件
                has_battery_pack = (item / "battery_pack.json").exists()
                has_bms_topology = (item / "bms_topology.json").exists()
                
                status = "[OK]" if (has_battery_pack and has_bms_topology) else "[ERROR]"
                folders.append(f"{status} {item.name}")
        
        if not folders:
            return TextContent(type="text", text="Batteries目录下没有找到任何文件夹")
        
        result = "可用的电池包文件夹:\n" + "\n".join(folders)
        result += "\n\n说明: [OK] 表示包含完整的配置文件，[ERROR] 表示缺少必需文件"
        
        return TextContent(type="text", text=result)
        
    except Exception as e:
        logger.error(f'列出文件夹时出错: {str(e)}')
        return TextContent(type="text", text=f"错误: {str(e)}")

@mcp.tool()
def predict_and_visualize_soh(npz_file_path: str, model_path: str) -> ImageContent:
    """使用指定模型对NPZ文件进行SOH预测并生成可视化图表
    
    Args:
        npz_file_path: NPZ文件的完整路径（支持原始格式和增强格式）
        model_path: 训练好的模型文件路径（.pth文件）
    
    Returns:
        包含SOH预测结果图表的PNG图片（base64编码）
    """
    temp_file = None
    
    try:
        logger.info(f'开始SOH预测: NPZ文件={npz_file_path}, 模型={model_path}')
        
        # 1. 检查文件是否存在
        if not os.path.exists(npz_file_path):
            raise FileNotFoundError(f"NPZ文件不存在: {npz_file_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 2. 检测NPZ文件格式
        file_format = detect_npz_format(npz_file_path)
        logger.info(f'检测到文件格式: {file_format}')
        
        # 3. 如果是原始格式，转换为增强格式
        if file_format == 'original':
            enhanced_file = convert_to_enhanced_format(npz_file_path)
            temp_file = enhanced_file  # 记录临时文件以便清理
        else:
            enhanced_file = npz_file_path
        
        # 4. 加载增强数据
        timeseries_data, true_soh_map, eol, num_cycles = load_enhanced_data(enhanced_file)
        logger.info(f'数据加载成功: {num_cycles} 个周期, EOL={eol}')
        
        # 5. 加载模型
        model = load_model_for_inference(model_path)
        
        # 6. 进行SOH预测
        predictions = predict_soh_for_visualization(model, timeseries_data)
        
        if not predictions:
            logger.warning("没有生成任何有效的预测结果")
            # 创建一个错误提示图片
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, '没有生成任何有效的预测结果\n请检查数据格式和模型兼容性', 
                       ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                error_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)
                buffer.close()
                
                return ImageContent(data=error_base64, mimeType="image/png", type="image")
            except:
                return ImageContent(data="", mimeType="image/png", type="image")
        
        # 7. 生成可视化图表
        battery_id = os.path.splitext(os.path.basename(npz_file_path))[0]
        image_base64 = create_soh_plot(predictions, battery_id)
        
        # 8. 输出统计信息到日志
        logger.info(f"预测统计:")
        logger.info(f"  预测周期范围: {min(predictions.keys())} - {max(predictions.keys())}")
        logger.info(f"  预测SOH范围: {min(predictions.values()):.4f} - {max(predictions.values()):.4f}")
        
        logger.info(f'成功生成SOH预测可视化图表: {battery_id}')
        return ImageContent(data=image_base64, mimeType="image/png", type="image")
        
    except Exception as e:
        logger.exception(f'SOH预测过程中出错: {str(e)}')
        
        # 创建一个错误信息图片
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, ax = plt.subplots(figsize=(10, 6))
            error_message = f'SOH预测出错:\n\n{str(e)[:200]}{"..." if len(str(e)) > 200 else ""}'
            ax.text(0.5, 0.5, error_message, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
                   wrap=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('SOH预测错误', fontsize=16, fontweight='bold')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            error_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            buffer.close()
            
            return ImageContent(data=error_base64, mimeType="image/png", type="image")
        except Exception as final_e:
            logger.error(f'创建错误图片时也失败了: {str(final_e)}')
            return ImageContent(data="", mimeType="image/png", type="image")
            
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"临时文件已清理: {temp_file}")
            except:
                logger.warning(f"警告: 无法删除临时文件: {temp_file}")

# 这是服务器的主入口点


@mcp.tool()
def visualize_battery_data(npz_file_path: str,
                          discharge: bool = True,
                          charge: bool = True,
                          cycle_numbers: List[List[int]] = [[-1, -1]],
                          x_col: int = 1,
                          y_col: int = 2,
                          save_path: str = "") -> ImageContent:
    """
    Illustration of charge/discharge curve of battery
     
    Args:
    npz_file_path (str): file path to .npz file of battery;
    discharge (bool): define whether to plot the discharge curve;
    charge (bool): define whether to plot the charge curve;
    cycle_numbers (array): An array like : [[the list of cycle numbers of discharging curve to be plotted], [the list of cycle numbers of charging curve to be ploted]]. Default is [-1,-1], it means to plot all the available cycles.
    x_col (int): The variable of x axis: 0: 'Parent Cycle Number', 1: 'Intra Process Cycle Fraction', 2: 'Voltage (V)' ,3: 'Current (A)',4: 'Capacity (Ah)',5: 'Relative Time (s)',6: 'Temperature (℃)',7: 'dQ/dV  (Ah/V)'.
    y_col (int): The variable of y axis: 0: 'Parent Cycle Number', 1: 'Intra Process Cycle Fraction', 2: 'Voltage (V)' ,3: 'Current (A)',4: 'Capacity (Ah)',5: 'Relative Time (s)',6: 'Temperature (℃)',7: 'dQ/dV  (Ah/V)'.
    save_path (str): the path to save the plotted figure. Default is '', the figure will be saved as 'figure1.jpg' at the same directory with the .npz file.
    
    Returns:
    Successfully execution: the base64 code to the plotted figure.
    Errors: the message of errors.
    """
    #check the input
    if isinstance(discharge, str):
        discharge=eval(discharge)
    if isinstance(charge, str):
        charge=eval(charge)  

    if isinstance(cycle_numbers, str):
        cycle_numbers=ast.literal_eval(cycle_numbers)
    if isinstance(x_col, str):
        x_col=int(x_col)    
    if isinstance(y_col, str):
        y_col=int(y_col)   
    
    try:
        # 加载数据
        with np.load(npz_file_path, allow_pickle=True) as data:
            metadata = data['metadata'].item()
            discharge_data = data['discharge_timeseries']
            charge_data = data['charge_timeseries']

    except Exception as e:
        return f"Fail to load the data: {str(e)}"            
            # 获取列名信息

    
    try:
        discharge_columns = {0: 'Parent Cycle Number', 1: 'Intra Process Cycle Fraction', 2: 'Voltage (V)' ,3: 'Current (A)',
                            4: 'Capacity (Ah)',
                            5: 'Relative Time (s)',
                            6: 'Temperature (℃)',
                            7: 'dQ/dV  (Ah/V)' }

            
        x_label = discharge_columns[x_col] if x_col < len(discharge_columns) else f"Column {x_col}"
        y_label = discharge_columns[y_col] if y_col < len(discharge_columns) else f"Column {y_col}"
        # 设置字体
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 24
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, min(len(np.unique(discharge_data[:, 0])), 
                                                     len(np.unique(charge_data[:, 0])))))
        
        # 处理放电曲线
        if discharge:
            discharge_cycles = np.unique(discharge_data[:, 0])
            if cycle_numbers[0] != -1:
                if isinstance(cycle_numbers[0], int):
                    discharge_cycles = [cycle_numbers[0], ]
                else:
                    discharge_cycles = [c for c in discharge_cycles if int(c) in cycle_numbers[0]]

            colors = plt.cm.autumn(np.linspace(0, 1,len(discharge_cycles)+1))
            
            for i, cycle in enumerate(discharge_cycles):
                cycle_mask = discharge_data[:, 0] == cycle
                cycle_data = discharge_data[cycle_mask]
                
                if len(cycle_data) > 0:
                    x_data = cycle_data[:, x_col]
                    y_data = cycle_data[:, y_col]
                    ax.plot(x_data, y_data, 
                           color=colors[i], 
                           linewidth=2, 
                           label=f'Discharge Cycle {int(cycle)}')
        
        # 处理充电曲线
        if charge:
            charge_cycles = np.unique(charge_data[:, 0])
            if cycle_numbers[1] != -1:
                if isinstance(cycle_numbers[1], int):
                    charge_cycles = [cycle_numbers[1], ]
                else:
                    charge_cycles = [c for c in charge_cycles if int(c) in cycle_numbers[1]]

            colors = plt.cm.winter(np.linspace(0, 1,len(charge_cycles)+1))
            for i, cycle in enumerate(charge_cycles):
                cycle_mask = charge_data[:, 0] == cycle
                cycle_data = charge_data[cycle_mask]
                
                if len(cycle_data) > 0:
                    x_data = cycle_data[:, x_col]
                    y_data = cycle_data[:, y_col]
                    ax.plot(x_data, y_data, 
                           color=colors[i], 
                           linewidth=2, 
                           linestyle='--',
                           label=f'Charge Cycle {int(cycle)}')
        
        # 设置标签和标题
        ax.set_xlabel(x_label, fontname='Arial', fontsize=24)
        ax.set_ylabel(y_label, fontname='Arial', fontsize=24)
        
        # 添加图例
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 确定保存路径
        if not save_path:
            npz_dir = os.path.dirname(npz_file_path)
            save_path = os.path.join(npz_dir, "figure1.jpg")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

        with open(save_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return ImageContent(data=encoded_string, mimeType="image/jpg", type="image")

        
    except Exception as e:
        error_msg = f"Fail to plot: {str(e)}\n{traceback.format_exc()}"
        return error_msg


@mcp.tool()
def plot_battery_curves_simple(npz_file_path: str,
                              curve_type: str = "both",
                              cycles_to_plot: str = "all",
                              x_axis: str = "voltage",
                              y_axis: str = "capacity",
                              legend_mode: str = "auto",
                              save_path: str = "") -> ImageContent:
    """
    简化版电池充放电曲线绘制函数

    Args:
        npz_file_path (str): NPZ文件路径
        curve_type (str): 曲线类型 - "discharge", "charge", "both"
        cycles_to_plot (str): 要绘制的周期 - "all", "first", "last", "1,5,10" (逗号分隔的周期号)
        x_axis (str): X轴变量 - "voltage", "capacity", "time", "fraction", "current", "temperature"
        y_axis (str): Y轴变量 - "voltage", "capacity", "time", "fraction", "current", "temperature"
        legend_mode (str): 图例显示模式 - "auto", "inside", "outside", "none", "compact"
        save_path (str): 保存路径，为空则自动生成

    Returns:
        ImageContent: 成功时返回图片的base64编码，失败时返回错误信息
    """
    try:
        # 变量映射字典
        var_mapping = {
            "cycle": 0,
            "fraction": 1, 
            "voltage": 2,
            "current": 3,
            "capacity": 4,
            "time": 5,
            "temperature": 6,
            "dqdv": 7
        }
        
        var_labels = {
            "cycle": "周期数",
            "fraction": "周期内进度",
            "voltage": "电压 (V)",
            "current": "电流 (A)", 
            "capacity": "容量 (Ah)",
            "time": "相对时间 (s)",
            "temperature": "温度 (℃)",
            "dqdv": "dQ/dV (Ah/V)"
        }
        
        # 获取列索引
        x_col = var_mapping.get(x_axis.lower(), 2)  # 默认电压
        y_col = var_mapping.get(y_axis.lower(), 4)  # 默认容量
        
        # 加载数据
        with np.load(npz_file_path, allow_pickle=True) as data:
            discharge_data = data['discharge_timeseries']
            charge_data = data['charge_timeseries']
        
        # 设置绘图样式
        plt.rcParams['font.family'] = 'SimHei'  # 中文字体
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 处理要绘制的周期
        def get_cycles_to_plot(data, cycles_spec):
            available_cycles = sorted(np.unique(data[:, 0]))
            if cycles_spec == "all":
                return available_cycles
            elif cycles_spec == "first":
                return [available_cycles[0]] if available_cycles else []
            elif cycles_spec == "last":
                return [available_cycles[-1]] if available_cycles else []
            else:
                # 解析逗号分隔的周期号
                try:
                    requested_cycles = [int(c.strip()) for c in cycles_spec.split(",")]
                    return [c for c in requested_cycles if c in available_cycles]
                except:
                    return available_cycles[:5]  # 默认前5个周期
        
        # 绘制放电曲线
        if curve_type in ["discharge", "both"]:
            discharge_cycles = get_cycles_to_plot(discharge_data, cycles_to_plot)
            colors_discharge = plt.cm.Reds(np.linspace(0.3, 1, len(discharge_cycles)))
            
            for i, cycle in enumerate(discharge_cycles):
                mask = discharge_data[:, 0] == cycle
                cycle_data = discharge_data[mask]
                if len(cycle_data) > 0:
                    x_data = cycle_data[:, x_col]
                    y_data = cycle_data[:, y_col]
                    ax.plot(x_data, y_data, 
                           color=colors_discharge[i],
                           linewidth=2,
                           label=f'放电周期 {int(cycle)}')
        
        # 绘制充电曲线
        if curve_type in ["charge", "both"]:
            charge_cycles = get_cycles_to_plot(charge_data, cycles_to_plot)
            colors_charge = plt.cm.Blues(np.linspace(0.3, 1, len(charge_cycles)))
            
            for i, cycle in enumerate(charge_cycles):
                mask = charge_data[:, 0] == cycle
                cycle_data = charge_data[mask]
                if len(cycle_data) > 0:
                    x_data = cycle_data[:, x_col]
                    y_data = cycle_data[:, y_col]
                    ax.plot(x_data, y_data,
                           color=colors_charge[i], 
                           linewidth=2,
                           linestyle='--',
                           label=f'充电周期 {int(cycle)}')
        
        # 设置标签
        ax.set_xlabel(var_labels.get(x_axis.lower(), x_axis))
        ax.set_ylabel(var_labels.get(y_axis.lower(), y_axis))
        ax.set_title('电池充放电曲线')

        # 添加图例和网格
        ax.grid(True, alpha=0.3)

        # 智能图例管理
        total_cycles = len(discharge_cycles) + len(charge_cycles) if curve_type == "both" else len(discharge_cycles if curve_type == "discharge" else charge_cycles)

        if legend_mode == "none":
            # 不显示图例
            pass
        elif legend_mode == "inside":
            # 图例显示在图内
            ax.legend(loc='best', fontsize=8)
        elif legend_mode == "compact":
            # 紧凑显示模式 - 当周期过多时分组显示
            if total_cycles > 10:
                # 创建分组标签
                handles, labels = ax.get_legend_handles_labels()
                if len(labels) > 10:
                    # 保留前5个和最后5个，中间显示省略号
                    new_handles = handles[:5] + handles[-5:]
                    new_labels = labels[:5] + ['...'] + labels[-5:]
                    ax.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                else:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        elif legend_mode == "outside":
            # 强制外部图例
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:  # auto mode
            # 自动模式 - 根据周期数量智能选择
            if total_cycles <= 5:
                # 少量周期时使用内部图例
                ax.legend(loc='best', fontsize=9)
            elif total_cycles <= 15:
                # 中等数量时使用外部图例
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                # 大量周期时使用紧凑模式
                handles, labels = ax.get_legend_handles_labels()
                if len(labels) > 15:
                    # 保留前5个和最后5个，中间显示省略号
                    new_handles = handles[:5] + handles[-5:]
                    new_labels = labels[:5] + ['...'] + labels[-5:]
                    ax.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
                else:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

        # 调整布局 - 根据图例模式调整
        if legend_mode in ["outside", "compact"] or (legend_mode == "auto" and total_cycles > 5):
            plt.tight_layout()
            # 为外部图例留出更多空间
            plt.subplots_adjust(right=0.8)
        else:
            plt.tight_layout()
        
        # 保存图片
        if not save_path:
            npz_dir = os.path.dirname(npz_file_path)
            save_path = os.path.join(npz_dir, "battery_curves_simple.jpg")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 返回base64编码
        with open(save_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return ImageContent(data=encoded_string, mimeType="image/jpeg", type="image")
            
    except Exception as e:
        error_msg = f"绘制失败: {str(e)}"
        return error_msg


@mcp.tool()
def plot_battery_curves_advanced(npz_file_path: str,
                                curve_type: str = "both",
                                cycles_to_plot: str = "all",
                                x_axis: str = "voltage",
                                y_axis: str = "capacity",
                                legend_mode: str = "auto",
                                max_legend_items: int = 20,
                                subplot_mode: bool = False,
                                save_path: str = "") -> ImageContent:
    """
    高级版电池充放电曲线绘制函数，支持复杂的legend管理

    Args:
        npz_file_path (str): NPZ文件路径
        curve_type (str): 曲线类型 - "discharge", "charge", "both"
        cycles_to_plot (str): 要绘制的周期 - "all", "first", "last", "1,5,10" (逗号分隔的周期号)
        x_axis (str): X轴变量 - "voltage", "capacity", "time", "fraction", "current", "temperature"
        y_axis (str): Y轴变量 - "voltage", "capacity", "time", "fraction", "current", "temperature"
        legend_mode (str): 图例显示模式 - "auto", "inside", "outside", "none", "compact", "separate"
        max_legend_items (int): 最大图例项目数量，超过时自动简化
        subplot_mode (bool): 是否使用子图模式显示多个周期
        save_path (str): 保存路径，为空则自动生成

    Returns:
        ImageContent: 成功时返回图片的base64编码，失败时返回错误信息
    """
    try:
        # 变量映射字典
        var_mapping = {
            "cycle": 0,
            "fraction": 1,
            "voltage": 2,
            "current": 3,
            "capacity": 4,
            "time": 5,
            "temperature": 6,
            "dqdv": 7
        }

        var_labels = {
            "cycle": "周期数",
            "fraction": "周期内进度",
            "voltage": "电压 (V)",
            "current": "电流 (A)",
            "capacity": "容量 (Ah)",
            "time": "相对时间 (s)",
            "temperature": "温度 (℃)",
            "dqdv": "dQ/dV (Ah/V)"
        }

        # 获取列索引
        x_col = var_mapping.get(x_axis.lower(), 2)
        y_col = var_mapping.get(y_axis.lower(), 4)

        # 加载数据
        with np.load(npz_file_path, allow_pickle=True) as data:
            discharge_data = data['discharge_timeseries']
            charge_data = data['charge_timeseries']

        # 设置绘图样式
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.unicode_minus'] = False

        # 处理要绘制的周期
        def get_cycles_to_plot(data, cycles_spec):
            available_cycles = sorted(np.unique(data[:, 0]))
            if cycles_spec == "all":
                return available_cycles
            elif cycles_spec == "first":
                return [available_cycles[0]] if available_cycles else []
            elif cycles_spec == "last":
                return [available_cycles[-1]] if available_cycles else []
            else:
                try:
                    requested_cycles = [int(c.strip()) for c in cycles_spec.split(",")]
                    return [c for c in requested_cycles if c in available_cycles]
                except:
                    return available_cycles[:5]

        discharge_cycles = get_cycles_to_plot(discharge_data, cycles_to_plot) if curve_type in ["discharge", "both"] else []
        charge_cycles = get_cycles_to_plot(charge_data, cycles_to_plot) if curve_type in ["charge", "both"] else []

        # 检查是否需要使用子图模式
        total_cycles = len(discharge_cycles) + len(charge_cycles)
        if subplot_mode or total_cycles > max_legend_items:
            # 子图模式 - 为每个周期创建单独的子图
            fig_rows = int(np.ceil(np.sqrt(total_cycles)))
            fig_cols = int(np.ceil(total_cycles / fig_rows))

            fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols*4, fig_rows*3))
            if fig_rows * fig_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            cycle_idx = 0
            all_handles = []
            all_labels = []

            # 绘制放电曲线
            for cycle in discharge_cycles:
                if cycle_idx >= len(axes):
                    break
                ax = axes[cycle_idx]
                mask = discharge_data[:, 0] == cycle
                cycle_data = discharge_data[mask]
                if len(cycle_data) > 0:
                    x_data = cycle_data[:, x_col]
                    y_data = cycle_data[:, y_col]
                    line, = ax.plot(x_data, y_data, 'r-', linewidth=2, label=f'放电周期 {int(cycle)}')
                    all_handles.append(line)
                    all_labels.append(f'放电周期 {int(cycle)}')

                    ax.set_xlabel(var_labels.get(x_axis.lower(), x_axis))
                    ax.set_ylabel(var_labels.get(y_axis.lower(), y_axis))
                    ax.set_title(f'周期 {int(cycle)}')
                    ax.grid(True, alpha=0.3)
                cycle_idx += 1

            # 绘制充电曲线
            for cycle in charge_cycles:
                if cycle_idx >= len(axes):
                    break
                ax = axes[cycle_idx]
                mask = charge_data[:, 0] == cycle
                cycle_data = charge_data[mask]
                if len(cycle_data) > 0:
                    x_data = cycle_data[:, x_col]
                    y_data = cycle_data[:, y_col]
                    line, = ax.plot(x_data, y_data, 'b--', linewidth=2, label=f'充电周期 {int(cycle)}')
                    all_handles.append(line)
                    all_labels.append(f'充电周期 {int(cycle)}')

                    ax.set_xlabel(var_labels.get(x_axis.lower(), x_axis))
                    ax.set_ylabel(var_labels.get(y_axis.lower(), y_axis))
                    ax.set_title(f'周期 {int(cycle)}')
                    ax.grid(True, alpha=0.3)
                cycle_idx += 1

            # 隐藏多余的子图
            for i in range(cycle_idx, len(axes)):
                axes[i].set_visible(False)

            # 如果需要图例且图例数量过多，创建单独的图例
            if legend_mode != "none" and len(all_handles) > 0:
                if len(all_handles) <= max_legend_items:
                    fig.legend(all_handles, all_labels, loc='center right', bbox_to_anchor=(1.05, 0.5))
                else:
                    # 创建简化图例
                    compact_handles = all_handles[:5] + all_handles[-5:] if len(all_handles) > 10 else all_handles
                    compact_labels = all_labels[:5] + ['...'] + all_labels[-5:] if len(all_labels) > 10 else all_labels
                    fig.legend(compact_handles, compact_labels, loc='center right', bbox_to_anchor=(1.05, 0.5), fontsize=8)

            plt.tight_layout()
            plt.subplots_adjust(right=0.85)

        else:
            # 普通模式
            fig, ax = plt.subplots(figsize=(12, 8))

            # 绘制放电曲线
            if discharge_cycles:
                colors_discharge = plt.cm.Reds(np.linspace(0.3, 1, len(discharge_cycles)))
                for i, cycle in enumerate(discharge_cycles):
                    mask = discharge_data[:, 0] == cycle
                    cycle_data = discharge_data[mask]
                    if len(cycle_data) > 0:
                        x_data = cycle_data[:, x_col]
                        y_data = cycle_data[:, y_col]
                        ax.plot(x_data, y_data,
                               color=colors_discharge[i],
                               linewidth=2,
                               label=f'放电周期 {int(cycle)}')

            # 绘制充电曲线
            if charge_cycles:
                colors_charge = plt.cm.Blues(np.linspace(0.3, 1, len(charge_cycles)))
                for i, cycle in enumerate(charge_cycles):
                    mask = charge_data[:, 0] == cycle
                    cycle_data = charge_data[mask]
                    if len(cycle_data) > 0:
                        x_data = cycle_data[:, x_col]
                        y_data = cycle_data[:, y_col]
                        ax.plot(x_data, y_data,
                               color=colors_charge[i],
                               linewidth=2,
                               linestyle='--',
                               label=f'充电周期 {int(cycle)}')

            # 设置标签
            ax.set_xlabel(var_labels.get(x_axis.lower(), x_axis))
            ax.set_ylabel(var_labels.get(y_axis.lower(), y_axis))
            ax.set_title('电池充放电曲线')
            ax.grid(True, alpha=0.3)

            # 智能图例管理
            handles, labels = ax.get_legend_handles_labels()

            if legend_mode == "none":
                pass
            elif legend_mode == "inside":
                ax.legend(loc='best', fontsize=8)
            elif legend_mode == "outside":
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            elif legend_mode == "separate":
                # 创建单独的图例面板
                if len(handles) > max_legend_items:
                    fig_legend = plt.figure(figsize=(8, 6))
                    compact_handles = handles[:max_legend_items//2] + handles[-(max_legend_items//2):]
                    compact_labels = labels[:max_legend_items//2] + ['...'] + labels[-(max_legend_items//2):]
                    fig_legend.legend(compact_handles, compact_labels, loc='center', fontsize=8)
                    fig_legend.savefig(save_path.replace('.jpg', '_legend.jpg'), dpi=300, bbox_inches='tight')
                    plt.close(fig_legend)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='部分图例\n(完整图例见单独文件)')
                else:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            elif legend_mode == "compact":
                if len(labels) > max_legend_items:
                    new_handles = handles[:max_legend_items//2] + handles[-(max_legend_items//2):]
                    new_labels = labels[:max_legend_items//2] + ['...'] + labels[-(max_legend_items//2):]
                    ax.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
                else:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:  # auto mode
                if len(labels) <= 5:
                    ax.legend(loc='best', fontsize=9)
                elif len(labels) <= max_legend_items:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                else:
                    new_handles = handles[:5] + handles[-5:]
                    new_labels = labels[:5] + ['...'] + labels[-5:]
                    ax.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

            plt.tight_layout()
            plt.subplots_adjust(right=0.8)

        # 保存图片
        if not save_path:
            npz_dir = os.path.dirname(npz_file_path)
            save_path = os.path.join(npz_dir, "battery_curves_advanced.jpg")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 返回base64编码
        with open(save_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return ImageContent(data=encoded_string, mimeType="image/jpeg", type="image")

    except Exception as e:
        error_msg = f"绘制失败: {str(e)}"
        return error_msg


@mcp.tool()
def extract_cell_status_info(json_file_path: str) -> TextContent:
    """
    Extract the cell ID and corresponding status information from the json file
    
    Args:
    json_file_path: file path of JSON file
    
    Returns:
    TextCotent: if succees return {CellID: dict of status}, else return the error message
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 提取电芯信息
        cell_status_dict = {}
        
        # 遍历所有的BMS单元
        for unit in data.get('bmsUnits', []):
            unit_id = unit.get('unitId')
            monitored_cells = unit.get('monitoredCellIds', [])
            status_info = unit.get('status', {})
            
            # 每个单元可能监控多个电芯
            for cell_id in monitored_cells:
                # 为每个电芯创建一个status信息的副本
                cell_status_dict[cell_id] = {
                    **status_info,  # 复制status信息
                    'monitoredBy': unit_id  # 添加监控单元信息
                }
        
        return TextContent(type="text", text=str(cell_status_dict))
        
    except FileNotFoundError:
        return TextContent(type="text",text=f"File is not founded: {json_file_path}")
    except json.JSONDecodeError:
        return TextContent(type="text",text=f"JSON file error: {json_file_path}")
    except Exception as e:
        raise TextContent(type="text",text=f"Fail to read the file: {str(e)}")


def main():
    logger.info('启动电池管理系统可视化MCP服务器')
    mcp.run('stdio')

if __name__ == "__main__":
    main()
