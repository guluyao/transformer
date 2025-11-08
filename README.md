
# Transformer期中作业：手工搭建与小规模文本建模实验

## 项目介绍
本项目基于PyTorch手工实现Transformer核心模块（Multi-Head Self-Attention、Position-wise FFN、残差+LayerNorm、位置编码等），并在Tiny Shakespeare数据集上完成语言建模任务及消融实验，验证关键模块的作用。

## 环境配置
### 1. 虚拟环境创建
```bash
conda create -n transformer python=3.10
conda activate transformer
```

### 2. 依赖安装
```bash
pip install -r requirements.txt
```

## 数据集说明
采用 **Tiny Shakespeare** 数据集（字符级语言建模，~1MB），用于验证Transformer在文本生成任务中的性能。
- 来源：[Hugging Face链接](https://huggingface.co/datasets/tiny_shakespeare)
- 存放路径：`datasets/tiny_shakespeare/input.txt`（已手动或通过脚本下载）

## 实验复现命令
### 1. 一键运行所有实验（基线模型+消融实验）
```bash
# Windows（CMD/PowerShell）
scripts\run.bat

# Linux/Mac 或 Windows Git Bash
chmod +x scripts/run.sh
scripts/run.sh
```
### 2. 一键分析所有实验结果
```bash
# Windows（CMD/PowerShell）
scripts\analyze.bat

# Linux/Mac 或 Windows Git Bash
chmod +x scripts/analyze.sh
scripts/analyze.sh
```
### 3. 单独运行基线模型（完整Transformer）
```bash
python src/train.py --d_model=128 --batch_size=32 
--block_size=64 --epochs=5 --learning_rate=3e-4 
--warmup_steps=2000 --result_dir=results/baseline/ 
--seed=42
```

### 消融位置编码
```bash
python src/train.py --d_model=128 --batch_size=32 
--block_size=64 --epochs=5 --learning_rate=3e-4 
--warmup_steps=2000 --result_dir=results/ablate_pe/ 
--ablate_pe --seed=42
```

###  消融单头注意力（消融Multi-Head）
```bash
python src/train.py --d_model=128 --batch_size=32 
--block_size=64 --epochs=5 --learning_rate=3e-4 --warmup_steps=2000 
--result_dir=results/ablate_multihead/ --ablate_multihead 
--seed=42
```
###  消融FFN
```bash
python src/train.py --d_model=128 --batch_size=32 
--block_size=64 --epochs=5 --learning_rate=3e-4 
--warmup_steps=2000 --result_dir=results/ablate_ffn/ --ablate_ffn 
--seed=42
```
## 硬件要求
- CPU：Intel i5及以上
- 内存：8GB及以上
- 无GPU依赖，纯CPU可运行

## 代码结构
```
transformer-midterm/
├── src/                # 源代码目录
│   ├── model.py        # Transformer核心模块（Multi-Head Attention、FFN等）
│   ├── train.py        # 训练与消融实验逻辑
│   ├── data.py         # 数据集加载与预处理
│   └── utils/          # 工具函数目录
│       └── utils.py    # 日志、可视化、结果保存工具
├── scripts/            # 运行脚本目录
│   ├── run.sh          # 一键运行所有实验（Linux/Mac/Git Bash）
├── results/            # 实验结果目录（训练曲线、表格、报告）
│   ├── baseline/       # 基线模型结果
│   ├── ablate_pe/      # 消融位置编码结果
│   ├── ablate_multihead/ # 消融多头注意力结果
│   └── ablate_ffn/     # 消融FFN结果
├── requirements.txt    # 依赖列表
└── README.md           # 项目说明
```


## 开源链接
GitHub仓库：(https://github.com/guluyao/transformer.git)
