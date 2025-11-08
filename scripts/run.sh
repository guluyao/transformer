#!/bin/bash
# ==============================================
# scripts/run.sh （Windows Git Bash/WSL 版）
# 需通过 Git Bash 或 WSL 运行
# ==============================================

# 通用训练参数
COMMON_PARAMS="--d_model=128 --batch_size=32 --block_size=64 --epochs=5 --learning_rate=3e-4 --warmup_steps=2000 --seed=42"

echo "=============================================="
echo "🔴 开始运行：基线模型（全组件）"
echo "=============================================="
python ../src/train.py $COMMON_PARAMS --result_dir=../results/baseline/
if [ $? -eq 0 ]; then
    echo "✅ 基线模型运行完成！结果保存在：../results/baseline/"
else
    echo "❌ 基线模型运行失败！"
    read -n 1 -s -r -p "按任意键退出..."
    exit 1
fi
echo ""

echo "=============================================="
echo "🔴 开始运行：消融位置编码"
echo "=============================================="
python ../src/train.py $COMMON_PARAMS --result_dir=../results/ablate_pe/ --ablate_pe
if [ $? -eq 0 ]; then
    echo "✅ 消融位置编码运行完成！结果保存在：../results/ablate_pe/"
else
    echo "❌ 消融位置编码运行失败！"
    read -n 1 -s -r -p "按任意键退出..."
    exit 1
fi
echo ""

echo "=============================================="
echo "🔴 开始运行：消融多头注意力（单头）"
echo "=============================================="
python ../src/train.py $COMMON_PARAMS --result_dir=../results/ablate_multihead/ --ablate_multihead
if [ $? -eq 0 ]; then
    echo "✅ 消融多头注意力运行完成！结果保存在：../results/ablate_multihead/"
else
    echo "❌ 消融多头注意力运行失败！"
    read -n 1 -s -r -p "按任意键退出..."
    exit 1
fi
echo ""

echo "=============================================="
echo "🔴 开始运行：消融 FFN"
echo "=============================================="
python ../src/train.py $COMMON_PARAMS --result_dir=../results/ablate_ffn/ --ablate_ffn
if [ $? -eq 0 ]; then
    echo "✅ 消融 FFN 运行完成！结果保存在：../results/ablate_ffn/"
else
    echo "❌ 消融 FFN 运行失败！"
    read -n 1 -s -r -p "按任意键退出..."
    exit 1
fi
echo ""

echo "=============================================="
echo "🎉 所有实验运行完毕！"
echo "下一步：执行 scripts/analyze.sh 分析结果"
echo "=============================================="
read -n 1 -s -r -p "按任意键退出..."