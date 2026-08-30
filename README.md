
# RT-DETRv2 + DINOv3 Detection Project

## 项目简介

本仓库是在 RT-DETRv2 的检测框架上，集成 DINOv3 Vision Transformer 作为特征提取器与多尺度编码器的目标检测项目。核心思路是：使用 DINOv3 作为强视觉主干，并从其中间 Transformer block 中提取多层语义特征；再通过层间融合与上/下采样构建三级特征金字塔，最终送入 RT-DETRv2 的解码器进行端到端目标检测。

该项目的主要特点包括：

- 使用 DINOv3 作为视觉编码器，强化语义表征能力；
- 采用多尺度特征金字塔设计，兼顾小目标检测和大目标语义建模；
- 保持 RT-DETRv2 的 Deformable DETR 风格解码器结构；
- 支持 COCO 训练、评估、单图推理和部署脚本；
- 适合研究 DINOv3 与 DETR 系列特征融合的实验场景。


## 目录结构

```text
.
├── configs/                  # 训练与实验配置
│   └── rtdetrv2/
├── ckpts/                    # 预训练权重目录
├── data/                     # 数据目录
├── dataset/                  # 数据集说明与工具
├── dinov3/                   # DINOv3 代码与加载逻辑
├── references/               # 推理/部署参考脚本
├── src/                      # 模型、训练、损失与后处理代码
├── tools/                    # 训练与分析脚本
├── Dockerfile                # 容器环境
├── docker-compose.yml        # 容器编排
├── requirements.txt          # Python 依赖
├── README.md                 # 项目说明

```


## 安装步骤

推荐使用 Conda 环境，当前仓库中有 Windows 兼容的 `.venv`，但在 Linux 环境中更推荐使用 `conda`。

### 1) 创建环境

```bash
conda create -n myDino python=3.10 -y
conda activate myDino
```

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

项目依赖包括：

- PyTorch / TorchVision
- PyYAML
- pycocotools
- scipy
- tensorboard
- transformers
- datasets
- onnx / onnxruntime

### 3) 准备预训练权重

将 DINOv3 的权重放入 `ckpts/`，例如：

```text
ckpts/
├── dinov3_vits16.pth
├── dinov3_vitb16.pth
├── dinov3_vitl16.pth
└── yolo26l.pt
```

### 4) 可选：使用 Docker

```bash
docker-compose up --build
```

或直接构建：

```bash
docker build -t rtdetr-v2:latest .
```


## 使用示例

### 1) 训练

训练入口来自 `tools/train.py`，示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --master_port=9909 --nproc_per_node=4 \
  tools/train.py -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco.yml \
  --use-amp --seed=0
```

也可以使用断点恢复：

```bash
CUDA_VISIBLE_DEVICES \
  python tools/train.py -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco.yml \
  -r path/to/checkpoint.pth --test-only
```

### 2) 测试 / 验证

```bash
CUDA_VISIBLE_DEVICES \
  torchrun --master_port  --nproc_per_node \
  tools/train.py -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco.yml \
  -r path/to/checkpoint.pth --test-only
```

### 3) 单图推理

项目提供了 PyTorch 推理脚本：

```bash
python references/deploy/rtdetrv2_torch.py \
  -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco.yml \
  -r path/to/checkpoint.pth \
  -f path/to/image.jpg \
  --device cuda:0 \
  --thresh 0.5
```

支持的可选参数包括：

- `--class-thresh`：按类别设置阈值；
- `--class-max-area-ratio`：按类别控制目标框面积比例；
- `--max-area-ratio`：全局最大面积比；
- `--suppress-containing-boxes`：抑制包含关系框。

### 4) 推理结果保存

推理完成后，脚本会在当前目录生成类似 `results_0.jpg` 的可视化结果图，包含：

- 检测框
- 类别名称
- 置信度标签
- 视觉化标注框和文本标签


## 技术栈说明

本项目主要技术栈如下：

- 目标检测框架：RT-DETRv2
- 视觉主干：DINOv3 Vision Transformer
- 深度学习框架：PyTorch
- 计算机视觉库：TorchVision
- 数据集支持：COCO
- 训练工具：torchrun + 自定义训练脚本
- 推理部署：PyTorch / ONNX / TensorRT 参考脚本
- 评估：pycocotools / faster-coco-eval

## 训练与配置说明

主要配置文件位于：

```text
configs/rtdetrv2/
├── rtdetrv2_dinov3_vit_6x_coco.yml
├── rtdetrv2_dinov3_vit_6x_coco_ETF.yml
└── ...
```



## 贡献指南（Contributing）

欢迎对本项目提出改进建议、修复错误和贡献实验结果。建议遵循以下流程：

1. Fork 本仓库；
2. 创建功能分支：

```bash
git checkout -b feature/my-improvement
```

3. 提交修改：

```bash
git add .
git commit -m "Add my improvement"
```

4. 推送分支：

```bash
git push origin feature/my-improvement
```

5. 在 GitHub 上提交 Pull Request，并说明：
   - 修改目的；
   - 相关配置或数据集；
   - 验证方式；
   - 是否影响现有训练流程。

建议在提交前：

- 确认配置文件正确；
- 保证模型加载与权重兼容；
- 对新增功能至少做一次最小验证；
- 在 PR 中描述实验结果和复现步骤。



> 说明：本 README 仅用于说明项目结构、安装和使用方式，不能替代正式的开源许可证声明。


## 参考信息

本项目的实现基于 RT-DETRv2 体系结构，并将 DINOv3 视觉主干接入到其多尺度检测流水线中。若你在研究中使用该代码，请保留相关论文和框架来源，并在实验报告中注明模型架构与预训练来源。


