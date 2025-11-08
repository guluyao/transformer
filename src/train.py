import math
import torch
import torch.nn as nn
import argparse
import os
import time
import logging
import numpy as np
from model import Transformer, create_pad_mask
from data import load_tiny_shakespeare
from utils.utils import setup_logger, get_optimizer_scheduler, clip_gradients, generate_comprehensive_plot, save_results_table

def train_epoch_complete(model, train_loader, criterion, optimizer, scheduler, device, logger, epoch_num, total_epochs):
    model.train()
    total_loss = 0.0
    total_batches = len(train_loader)
    processed_samples = 0
    logger.info(f"Epoch {epoch_num}/{total_epochs}: 开始训练 ({total_batches:,} 批次)")
    start_time = time.time()
    batch_times = []
    for batch_idx, (x, y) in enumerate(train_loader):
        batch_start = time.time()
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        processed_samples += batch_size
        try:
            src_mask = create_pad_mask(x)
            tgt_input = x[:, :-1]
            outputs = model(x, tgt_input, src_mask, None)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), y[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            clip_gradients(model, max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * batch_size
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            log_interval = max(1, total_batches // 100)
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                avg_loss = total_loss / processed_samples
                progress = (batch_idx + 1) / total_batches * 100
                current_lr = optimizer.param_groups[0]['lr']
                elapsed_time = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed_time
                remaining_batches = total_batches - (batch_idx + 1)
                eta_seconds = remaining_batches / batches_per_sec if batches_per_sec > 0 else 0
                recent_avg_time = np.mean(batch_times[-10:]) if len(batch_times) >= 10 else batch_time
                logger.info(
                    f"进度: {progress:5.1f}% | 批次: {batch_idx + 1:,}/{total_batches:,} | 损失: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | 速度: {recent_avg_time:.3f}s/batch | ETA: {eta_seconds / 60:.1f}min"
                )
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"批次 {batch_idx + 1} 内存不足，跳过")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"批次 {batch_idx + 1} 运行时错误: {e}")
                continue
        except Exception as e:
            logger.error(f"批次 {batch_idx + 1} 未知错误: {e}")
            continue
    epoch_time = time.time() - start_time
    avg_epoch_loss = total_loss / processed_samples if processed_samples > 0 else float('inf')
    logger.info(
        f"Epoch {epoch_num} 完成! | 平均损失: {avg_epoch_loss:.4f} | 耗时: {epoch_time / 60:.1f}分钟 | "
        f"处理样本: {processed_samples:,} | 总批次: {total_batches:,}"
    )
    return avg_epoch_loss

def validate_epoch_complete(model, val_loader, criterion, device, logger):
    model.eval()
    total_loss = 0.0
    processed_samples = 0
    logger.info("开始验证...")
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            processed_samples += batch_size
            src_mask = create_pad_mask(x)
            tgt_input = x[:, :-1]
            outputs = model(x, tgt_input, src_mask, None)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), y[:, 1:].reshape(-1))
            total_loss += loss.item() * batch_size
    val_time = time.time() - start_time
    avg_val_loss = total_loss / processed_samples
    val_perplexity = math.exp(avg_val_loss)
    logger.info(
        f"验证完成! | 损失: {avg_val_loss:.4f} | 困惑度: {val_perplexity:.2f} | 耗时: {val_time:.1f}秒 | "
        f"样本数: {processed_samples:,}"
    )
    return avg_val_loss, val_perplexity

def main_complete():
    parser = argparse.ArgumentParser(description="Transformer完整数据训练")
    parser.add_argument("--d_model", type=int, default=128, help="模型维度")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮次")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="预热步数")
    parser.add_argument("--result_dir", type=str, default="results/complete/", help="实验结果保存目录")
    parser.add_argument("--skip_validation", action="store_true", help="跳过验证")
    # 所有消融实验参数（含之前的pe、multihead，新增ffn）
    parser.add_argument("--ablate_pe", action="store_true", help="消融位置编码")
    parser.add_argument("--ablate_multihead", action="store_true", help="消融多头注意力（使用单头）")
    parser.add_argument("--ablate_ffn", action="store_true", help="消融逐位置前馈网络（FFN）")  # 新增FFN消融参数
    parser.add_argument("--block_size", type=int, default=128, help="序列长度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)
    logger = setup_logger(args.result_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info(f"实验配置: d_model={args.d_model}, batch_size={args.batch_size}, epochs={args.epochs}, "
                f"block_size={args.block_size}, seed={args.seed}, ablate_pe={args.ablate_pe}, "
                f"ablate_multihead={args.ablate_multihead}, ablate_ffn={args.ablate_ffn}")  # 日志打印FFN消融状态

    try:
        logger.info("加载数据集...")
        train_loader, val_loader, vocab_size, stoi, itos = load_tiny_shakespeare(block_size=args.block_size)
        logger.info(f"训练批次: {len(train_loader):,}, 验证批次: {len(val_loader):,}, 词汇表大小: {vocab_size}")

        logger.info("初始化模型...")
        model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=args.d_model,
            n_enc_layers=2,
            n_dec_layers=2,
            h=2,
            d_ff=256,
            dropout=0.1,
            max_seq_len=args.block_size,
            ablate_pe=args.ablate_pe,
            ablate_multihead=args.ablate_multihead,
            ablate_ffn=args.ablate_ffn  # 传递FFN消融参数给模型
        ).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer, scheduler = get_optimizer_scheduler(
            model,
            d_model=args.d_model,
            warmup_steps=args.warmup_steps,
            lr=args.learning_rate
        )

        train_losses = []
        val_losses = []
        val_perplexities = []
        start_time = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")
            logger.info(f"{'=' * 60}")

            train_loss = train_epoch_complete(
                model, train_loader, criterion, optimizer, scheduler,
                device, logger, epoch + 1, args.epochs
            )
            train_losses.append(train_loss)

            if not args.skip_validation and len(val_loader) > 0:
                val_loss, val_perp = validate_epoch_complete(
                    model, val_loader, criterion, device, logger
                )
                val_losses.append(val_loss)
                val_perplexities.append(val_perp)
            else:
                val_losses.append(None)
                val_perplexities.append(None)

            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1} 统计: 训练损失={train_loss:.4f}, "
                        f"验证损失={val_losses[-1] if not args.skip_validation else '跳过'}, "
                        f"耗时={epoch_time / 60:.1f}分钟, 累计耗时={total_time / 60:.1f}分钟")

            results = []
            for i in range(len(train_losses)):
                results.append({
                    "epoch": i + 1,
                    "train_loss": train_losses[i],
                    "val_loss": val_losses[i] if i < len(val_losses) else None,
                    "val_perplexity": val_perplexities[i] if i < len(val_perplexities) else None
                })
            save_results_table(results, os.path.join(args.result_dir, "training_results.csv"))
            generate_comprehensive_plot(train_losses, val_losses, val_perplexities,
                                        os.path.join(args.result_dir, "training_curve.png"))
            logger.info(f"结果已保存至 {args.result_dir}")

        total_time = time.time() - start_time
        logger.info(f"\n训练完成! 总耗时: {total_time / 60:.1f}分钟, 最终训练损失: {train_losses[-1]:.4f}")
        if not args.skip_validation:
            logger.info(f"最终验证损失: {val_losses[-1]:.4f}, 困惑度: {val_perplexities[-1]:.2f}")

        report_path = os.path.join(args.result_dir, "training_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Transformer语言建模实验报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总epoch数: {len(train_losses)}\n")
            f.write(f"最终训练损失: {train_losses[-1]:.4f}\n")
            f.write(f"总耗时: {total_time / 60:.1f}分钟\n\n")
            f.write("训练配置:\n")
            f.write(f"- 模型维度: {args.d_model}\n")
            f.write(f"- 批次大小: {args.batch_size}\n")
            f.write(f"- 学习率: {args.learning_rate}\n")
            f.write(f"- 训练轮次: {args.epochs}\n")
            f.write(f"- 序列长度: {args.block_size}\n")
            f.write(f"- 随机种子: {args.seed}\n")
            f.write(f"- 是否消融位置编码: {'是' if args.ablate_pe else '否'}\n")
            f.write(f"- 是否消融多头注意力: {'是' if args.ablate_multihead else '否'}\n")
            f.write(f"- 是否消融FFN: {'是' if args.ablate_ffn else '否'}\n\n")  # 报告新增FFN消融状态
            f.write("详细结果:\n")
            for i, loss in enumerate(train_losses):
                f.write(f"Epoch {i + 1}: 训练损失={loss:.4f}")
                if not args.skip_validation and i < len(val_losses):
                    f.write(f", 验证损失={val_losses[i]:.4f}, 困惑度={val_perplexities[i]:.2f}")
                f.write("\n")
        logger.info(f"实验报告已保存至 {report_path}")

    except Exception as e:
        logger.error(f"训练错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main_complete()