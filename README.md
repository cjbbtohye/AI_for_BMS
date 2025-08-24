# 电池管理系统可视化 MCP 服务器

这是一个基于 FastMCP 的电池管理系统可视化服务器，可以读取电池包配置和BMS拓扑数据，生成美观的可视化图表。

## 功能特性

- 📊 **BMS拓扑可视化**: 展示BMS节点的连接关系和层次结构
- 🔗 **多种连接方式**: 支持CAN总线、菊花链等不同连接协议
- 🖼️ **Base64图片输出**: 直接返回可在聊天界面显示的图片
- 📁 **多电池包支持**: 可处理多个不同的电池包配置
- 🎨 **智能角色识别**: 自动识别并可视化不同类型的BMS节点（主控、从控、监控节点等）
- 🌐 **外部接口展示**: 显示中央控制器的外部通信接口

## 安装依赖

### 1. 安装Python依赖
```bash
uv sync
```

### 2. 安装Graphviz可执行文件

本项目使用Graphviz生成可视化图表，需要安装Graphviz可执行文件：

#### Windows用户：
1. 从 [Graphviz官网](https://graphviz.org/download/) 下载Windows安装包
2. 运行安装程序，选择"Add Graphviz to the system PATH"选项
3. 重启命令行窗口
4. 验证安装：`dot -V`

#### 使用Chocolatey（推荐）：
```bash
choco install graphviz
```

#### 使用Scoop：
```bash
scoop install graphviz
```

#### macOS用户：
```bash
brew install graphviz
```

#### Linux用户：
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# CentOS/RHEL
sudo yum install graphviz

# Arch Linux
sudo pacman -S graphviz
```

### 3. 验证安装
安装完成后，运行以下命令验证：
```bash
dot -V
python -c "import graphviz; print('Graphviz Python库已安装')"
```

## MCP工具函数

### 1. `visualize_battery_system(folder_name)`
生成指定电池包文件夹的可视化图表。

**参数**:
- `folder_name`: Batteries目录下的文件夹名称，必须包含 `battery_pack.json` 和 `bms_topology.json` 文件

**返回**: PNG格式的图片（base64编码），显示BMS拓扑结构图，包含：
- 📦 电池包基本信息（名称、配置、电压、容量等）
- 🎛️ BMS节点的角色和类型（主控、从控、监控节点等）
- 🔗 节点间的连接关系（CAN总线、菊花链等）
- 🔌 外部接口信息（以太网、干接点、模拟量等）

### 2. `list_available_battery_folders()`
列出所有可用的电池包文件夹。

**返回**: 文本列表，显示哪些文件夹包含完整的配置文件

## 数据格式

### 电池包配置文件 (battery_pack.json)
包含电池包的基本信息、规格参数、电池单元规格等：

```json
{
  "packInfo": {
    "packID": "BP-4S2P-001",
    "packName": "4S2P Example Pack",
    ...
  },
  "configuration": {
    "seriesCount": 4,
    "parallelCount": 2,
    "totalCells": 8,
    "arrangement": "4S2P"
  },
  ...
}
```

### BMS拓扑文件 (bms_topology.json)
包含BMS节点的连接信息：

```json
{
  "bmsUnits": [
    {
      "unitId": "Node-01",
      "role": "node",
      "model": "D-NODE-V1",
      "monitoredCellIds": ["C01"],
      "connections": {
        "protocol": "ISOSPI",
        "daisyChainNext": "Node-02"
      }
    },
    ...
  ]
}
```

## 运行服务器

```bash
python main.py
```

## 示例用法

1. 列出可用的电池包：
   ```
   调用 list_available_battery_folders()
   ```

2. 生成4S2P示例的可视化图表：
   ```
   调用 visualize_battery_system("4S2P_example")
   ```

## 图表功能

生成的可视化图表包含：

### 📋 顶部标题区
- 📦 电池包名称和制造商信息
- ⚡ 串并联配置（如4S2P）
- 🔋 电压和容量规格

### 🎛️ BMS节点可视化
- **主控/中央控制器**: 蓝色/绿色矩形，显示控制功能
- **从控/监控节点**: 橙色/红色椭圆，显示监控范围
- **节点信息**: 型号、监控电芯数量、特殊功能

### 🔗 连接关系
- **CAN总线**: 蓝色实线，表示控制连接
- **菊花链**: 红色虚线，显示协议类型（如ISOSPI）
- **外部接口**: 绿色点线，连接到接口说明

### 🔌 外部接口
- 以太网、RS485等通信接口
- 干接点、模拟量等信号接口
- 上位机监控、PCS接口等用途说明

## 文件结构

```
BMS_2/
├── main.py                    # MCP服务器主文件
├── pyproject.toml            # 项目配置和依赖
└── Batteries/                # 电池包数据目录
    └── 4S2P_example/         # 示例电池包
        ├── battery_pack.json # 电池包配置
        └── bms_topology.json # BMS拓扑
```

## 技术特点

- 🚀 基于 FastMCP 框架
- 📈 使用 Graphviz 生成专业级拓扑图
- 🎨 支持中文显示和表情符号
- 💾 自动Base64编码，便于传输
- 🔧 模块化设计，易于扩展
- 🌐 智能数据结构适配，支持多种JSON格式
- 🎯 角色驱动的可视化，不同节点类型自动使用不同样式
