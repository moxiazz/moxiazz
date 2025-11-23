# Chaotic Cold Diffusion Framework

[中文](#中文说明) | [English](#english-description)

---

## 中文说明

### 概述

这是一个基于 **Cold Diffusion** 框架的混沌序列生成和建模系统。Cold Diffusion 是传统扩散模型的泛化，它不仅限于添加高斯噪声，而是可以使用任意的退化算子。本框架专门设计用于处理和生成混沌序列。

### 核心特性

- **多种混沌系统**：支持 Logistic Map、Henon Map、Lorenz System 等经典混沌系统
- **多样化的退化算子**：
  - 基于噪声的退化（线性/余弦调度）
  - 排列退化（破坏时间相关性）
  - 掩码退化（类似于掩码语言模型）
- **Cold Diffusion 框架**：适用于混沌序列的扩散模型实现
- **可视化工具**：用于序列、吸引子和扩散过程的可视化

### 项目结构

```
chaotic_diffusion/
├── core/                      # 核心组件
│   ├── __init__.py
│   └── chaotic_systems.py     # 混沌系统实现
├── models/                    # 模型实现
│   ├── __init__.py
│   ├── cold_diffusion.py      # Cold Diffusion 模型
│   └── degradation.py         # 退化算子
├── utils/                     # 工具函数
│   ├── __init__.py
│   └── visualization.py       # 可视化工具
└── examples/                  # 示例脚本
    ├── basic_example.py       # 基础示例
    └── advanced_example.py    # 高级示例
```

### 快速开始

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 基础示例

```python
from chaotic_diffusion.core import LogisticMap
from chaotic_diffusion.models import ChaoticColdDiffusion, ChaoticDegradation

# 1. 生成混沌序列
logistic = LogisticMap(r=3.9)
sequence = logistic.generate_sequence(length=200, initial_state=0.1)

# 2. 创建 Cold Diffusion 模型
degradation = ChaoticDegradation(noise_schedule="linear")
diffusion = ChaoticColdDiffusion(
    degradation=degradation,
    T=100,
    sequence_length=200,
    sequence_dim=1
)

# 3. 前向扩散过程
degraded = diffusion.forward_process(sequence, t=50)

# 4. 反向采样生成新序列
generated = diffusion.sample()
```

#### 运行示例

```bash
# 基础示例
python chaotic_diffusion/examples/basic_example.py

# 高级示例（多种混沌系统和退化算子）
python chaotic_diffusion/examples/advanced_example.py
```

### 支持的混沌系统

1. **Logistic Map**：一维混沌映射
   - 方程：`x_{n+1} = r * x_n * (1 - x_n)`
   - 参数范围：`r ∈ [3.57, 4.0]` 时表现混沌行为

2. **Henon Map**：二维混沌映射
   - 方程：
     - `x_{n+1} = 1 - a * x_n^2 + y_n`
     - `y_{n+1} = b * x_n`
   - 默认参数：`a=1.4, b=0.3`

3. **Lorenz System**：三维混沌系统
   - 方程：
     - `dx/dt = σ(y - x)`
     - `dy/dt = x(ρ - z) - y`
     - `dz/dt = xy - βz`
   - 默认参数：`σ=10, ρ=28, β=8/3`

### Cold Diffusion 原理

Cold Diffusion 是对传统扩散模型的推广：

1. **传统扩散**：逐步添加高斯噪声
2. **Cold Diffusion**：使用任意退化算子

**前向过程**（退化）：
- 将干净的混沌序列逐步退化
- 退化可以是噪声、平滑、排列等

**反向过程**（生成）：
- 从完全退化的状态开始
- 逐步恢复/生成混沌序列
- 使用训练的模型预测恢复方向

### 退化算子类型

1. **ChaoticDegradation**：
   - 逐步添加噪声
   - 平滑处理（破坏细粒度混沌结构）
   - 支持线性和余弦噪声调度

2. **PermutationDegradation**：
   - 逐步打乱序列顺序
   - 破坏时间相关性

3. **MaskingDegradation**：
   - 逐步掩码/移除元素
   - 类似于掩码语言模型

### 技术细节

- **扩散步数**：默认 T=1000，可根据需要调整
- **噪声调度**：支持线性和余弦调度
- **模型架构**：提供简单 MLP 实现，可替换为更复杂的网络（U-Net、Transformer 等）

### 扩展和自定义

您可以通过以下方式扩展框架：

1. **添加新的混沌系统**：继承 `ChaoticSystem` 类
2. **实现新的退化算子**：继承 `DegradationOperator` 类
3. **使用深度学习模型**：替换 `NeuralRestorationModel` 为 PyTorch/TensorFlow 模型

### 应用场景

- 混沌时间序列生成
- 混沌系统建模
- 异常检测（利用重构误差）
- 数据增强
- 序列去噪

---

## English Description

### Overview

This is a **Cold Diffusion**-based framework for chaotic sequence generation and modeling. Cold Diffusion generalizes traditional diffusion models by using arbitrary degradation operators instead of just Gaussian noise. This framework is specifically designed for chaotic sequences.

### Key Features

- **Multiple Chaotic Systems**: Supports classic chaotic systems like Logistic Map, Henon Map, and Lorenz System
- **Diverse Degradation Operators**:
  - Noise-based degradation (linear/cosine schedules)
  - Permutation degradation (destroys temporal correlations)
  - Masking degradation (similar to masked language modeling)
- **Cold Diffusion Framework**: Diffusion model implementation tailored for chaotic sequences
- **Visualization Tools**: For sequences, attractors, and diffusion processes

### Project Structure

```
chaotic_diffusion/
├── core/                      # Core components
│   ├── __init__.py
│   └── chaotic_systems.py     # Chaotic systems implementation
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── cold_diffusion.py      # Cold Diffusion model
│   └── degradation.py         # Degradation operators
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── visualization.py       # Visualization tools
└── examples/                  # Example scripts
    ├── basic_example.py       # Basic example
    └── advanced_example.py    # Advanced example
```

### Quick Start

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Basic Example

```python
from chaotic_diffusion.core import LogisticMap
from chaotic_diffusion.models import ChaoticColdDiffusion, ChaoticDegradation

# 1. Generate chaotic sequence
logistic = LogisticMap(r=3.9)
sequence = logistic.generate_sequence(length=200, initial_state=0.1)

# 2. Create Cold Diffusion model
degradation = ChaoticDegradation(noise_schedule="linear")
diffusion = ChaoticColdDiffusion(
    degradation=degradation,
    T=100,
    sequence_length=200,
    sequence_dim=1
)

# 3. Forward diffusion process
degraded = diffusion.forward_process(sequence, t=50)

# 4. Sample/generate new sequence
generated = diffusion.sample()
```

#### Run Examples

```bash
# Basic example
python chaotic_diffusion/examples/basic_example.py

# Advanced example (multiple systems and degradations)
python chaotic_diffusion/examples/advanced_example.py
```

### Supported Chaotic Systems

1. **Logistic Map**: 1D chaotic map
   - Equation: `x_{n+1} = r * x_n * (1 - x_n)`
   - Chaotic for: `r ∈ [3.57, 4.0]`

2. **Henon Map**: 2D chaotic map
   - Equations:
     - `x_{n+1} = 1 - a * x_n^2 + y_n`
     - `y_{n+1} = b * x_n`
   - Default: `a=1.4, b=0.3`

3. **Lorenz System**: 3D chaotic system
   - Equations:
     - `dx/dt = σ(y - x)`
     - `dy/dt = x(ρ - z) - y`
     - `dz/dt = xy - βz`
   - Default: `σ=10, ρ=28, β=8/3`

### Cold Diffusion Principles

Cold Diffusion generalizes traditional diffusion models:

1. **Traditional Diffusion**: Progressively adds Gaussian noise
2. **Cold Diffusion**: Uses arbitrary degradation operators

**Forward Process** (Degradation):
- Progressively degrades clean chaotic sequences
- Degradation can be noise, smoothing, permutation, etc.

**Reverse Process** (Generation):
- Starts from fully degraded state
- Progressively restores/generates chaotic sequence
- Uses trained model to predict restoration direction

### Degradation Operator Types

1. **ChaoticDegradation**:
   - Progressively adds noise
   - Applies smoothing (destroys fine-scale chaotic structure)
   - Supports linear and cosine noise schedules

2. **PermutationDegradation**:
   - Progressively shuffles sequence order
   - Destroys temporal correlations

3. **MaskingDegradation**:
   - Progressively masks/removes elements
   - Similar to masked language modeling

### Technical Details

- **Diffusion Steps**: Default T=1000, adjustable as needed
- **Noise Schedules**: Supports linear and cosine schedules
- **Model Architecture**: Provides simple MLP implementation, can be replaced with more sophisticated networks (U-Net, Transformer, etc.)

### Extension and Customization

You can extend the framework by:

1. **Adding new chaotic systems**: Inherit from `ChaoticSystem` class
2. **Implementing new degradation operators**: Inherit from `DegradationOperator` class
3. **Using deep learning models**: Replace `NeuralRestorationModel` with PyTorch/TensorFlow models

### Applications

- Chaotic time series generation
- Chaotic system modeling
- Anomaly detection (using reconstruction error)
- Data augmentation
- Sequence denoising

### Citation

If you use this framework in your research, please cite:

```bibtex
@software{chaotic_cold_diffusion,
  title = {Chaotic Cold Diffusion Framework},
  author = {moxiazz},
  year = {2025},
  url = {https://github.com/moxiazz/moxiazz}
}
```

### License

This project is open source and available under the MIT License.

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Contact

For questions and feedback, please open an issue on GitHub.
