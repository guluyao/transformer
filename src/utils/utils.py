import torch
import math
import logging
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def setup_logger(result_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(console_handler)
    log_path = os.path.join(result_dir, "training.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

def get_optimizer_scheduler(model, d_model=128, warmup_steps=4000, lr=3e-4):
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    def lr_lambda(step):
        step = max(1, step)
        return math.pow(d_model, -0.5) * min(
            math.pow(step, -0.5),
            step * math.pow(warmup_steps, -1.5)
        )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

def clip_gradients(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

def save_model(model, optimizer, scheduler, epoch, loss, save_path="checkpoints/transformer_epoch{}.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": loss
    }, save_path.format(epoch))

def load_model(model, optimizer, scheduler, load_path):
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return model, optimizer, scheduler, checkpoint["epoch"], checkpoint["val_loss"]

def generate_comprehensive_plot(train_losses, val_losses=None, val_perplexities=None,
                                save_path="results/training_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        epochs = range(1, len(train_losses) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title('Training Loss (Linear Scale)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(epochs, train_losses, 'r-', linewidth=2, marker='s', markersize=4)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Training Loss')
        axes[0, 1].set_title('Training Loss (Log Scale)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        if val_losses and any(loss is not None for loss in val_losses):
            valid_epochs = [epoch for epoch, loss in zip(epochs, val_losses) if loss is not None]
            valid_losses = [loss for loss in val_losses if loss is not None]
            axes[1, 0].plot(valid_epochs, valid_losses, 'g-', linewidth=2, marker='^', markersize=4)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Validation Loss')
            axes[1, 0].set_title('Validation Loss')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Validation Loss (No Data)')
        if val_perplexities and any(ppl is not None for ppl in val_perplexities):
            valid_epochs = [epoch for epoch, ppl in zip(epochs, val_perplexities) if ppl is not None]
            valid_ppls = [ppl for ppl in val_perplexities if ppl is not None]
            axes[1, 1].plot(valid_epochs, valid_ppls, 'purple', linewidth=2, marker='d', markersize=4)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Perplexity')
            axes[1, 1].set_title('Validation Perplexity')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Validation Perplexity (No Data)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        logging.error(f"生成图表出错: {e}")
        return False

def save_results_table(results, save_path="results/experiment_results.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False, encoding="utf-8")

def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_million": round(total_params / 1e6, 2),
        "trainable_params_million": round(trainable_params / 1e6, 2)
    }