import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import sys
import os

# 添加当前目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import build_poisoned_training_set, build_testset
from torch.utils.data import DataLoader

def visualize_samples(dataset_clean, dataset_poisoned, dataset_name, trigger_label, num_samples=5):
    """
    可视化干净样本和对应的投毒样本
    """
    # 设置中文字体（如果需要显示中文标签）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    # 获取类别名称
    if hasattr(dataset_clean, 'classes'):
        class_names = dataset_clean.classes
    else:
        class_names = [str(i) for i in range(10)]  # 假设10个类别
    
    for i in range(num_samples):
        # 获取干净样本
        clean_img, clean_label = dataset_clean[i]
        
        # 获取对应的投毒样本
        poison_idx = i % len(dataset_poisoned)
        poison_img, poison_label = dataset_poisoned[poison_idx]
        
        # 转换张量格式以便显示
        # PyTorch张量通常是 (C, H, W)，需要转换为 (H, W, C) 用于matplotlib
        if isinstance(clean_img, torch.Tensor):
            clean_img_np = clean_img.numpy().transpose(1, 2, 0)
            poison_img_np = poison_img.numpy().transpose(1, 2, 0)
        else:
            clean_img_np = clean_img
            poison_img_np = poison_img
        
        # 对于单通道图像，去除通道维度
        if clean_img_np.shape[-1] == 1:
            clean_img_np = clean_img_np.squeeze(-1)
            poison_img_np = poison_img_np.squeeze(-1)
        
        # 显示干净样本
        axes[0, i].imshow(clean_img_np, cmap='gray' if len(clean_img_np.shape) == 2 else None)
        axes[0, i].set_title(f'Clean Sample\nLabel: {class_names[clean_label]}')
        axes[0, i].axis('off')
        
        # 显示投毒样本
        axes[1, i].imshow(poison_img_np, cmap='gray' if len(poison_img_np.shape) == 2 else None)
        axes[1, i].set_title(f'Poisoned Sample\nLabel: {class_names[poison_label]}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Dataset: {dataset_name} | Target Label: {trigger_label}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'./samples_comparison_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_single_comparison(dataset_clean, dataset_poisoned, dataset_name, trigger_label, index=0):
    """
    可视化单个样本的干净版本和投毒版本对比
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # 获取类别名称
    if hasattr(dataset_clean, 'classes'):
        class_names = dataset_clean.classes
    else:
        class_names = [str(i) for i in range(10)]
    
    # 获取干净样本
    clean_img, clean_label = dataset_clean[index]
    
    # 获取投毒样本
    poison_idx = index % len(dataset_poisoned)
    poison_img, poison_label = dataset_poisoned[poison_idx]
    
    # 转换张量格式
    if isinstance(clean_img, torch.Tensor):
        clean_img_np = clean_img.numpy().transpose(1, 2, 0)
        poison_img_np = poison_img.numpy().transpose(1, 2, 0)
    else:
        clean_img_np = clean_img
        poison_img_np = poison_img
    
    # 对于单通道图像，去除通道维度
    if clean_img_np.shape[-1] == 1:
        clean_img_np = clean_img_np.squeeze(-1)
        poison_img_np = poison_img_np.squeeze(-1)
    
    # 显示
    ax1.imshow(clean_img_np, cmap='gray' if len(clean_img_np.shape) == 2 else None)
    ax1.set_title(f'Clean Sample\nTrue Label: {class_names[clean_label]}')
    ax1.axis('off')
    
    ax2.imshow(poison_img_np, cmap='gray' if len(poison_img_np.shape) == 2 else None)
    ax2.set_title(f'Poisoned Sample\nTarget Label: {class_names[poison_label]}')
    ax2.axis('off')
    
    plt.suptitle(f'Dataset: {dataset_name} | Target Label: {trigger_label}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'./single_comparison_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"干净样本信息:")
    print(f"  - 索引: {index}")
    print(f"  - 原始标签: {clean_label} ({class_names[clean_label]})")
    print(f"  - 图像形状: {clean_img_np.shape}")
    print(f"投毒样本信息:")
    print(f"  - 索引: {poison_idx}")
    print(f"  - 目标标签: {poison_label} ({class_names[poison_label]})")
    print(f"  - 图像形状: {poison_img_np.shape}")

def main():
    parser = argparse.ArgumentParser(description='Visualize clean and poisoned samples')
    parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
    parser.add_argument('--trigger_label', type=int, default=1, help='The target label for poisoning')
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help='Poisoning rate')
    parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger path')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger size')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # 创建模拟的args对象，与main.py兼容
    class Args:
        pass
    
    visual_args = Args()
    visual_args.dataset = args.dataset
    visual_args.trigger_label = args.trigger_label
    visual_args.poisoning_rate = args.poisoning_rate
    visual_args.trigger_path = args.trigger_path
    visual_args.trigger_size = args.trigger_size
    visual_args.download = True  # 允许下载数据
    visual_args.data_path = './data/'
    
    print("Loading datasets...")
    try:
        # 构建数据集
        dataset_train, nb_classes = build_poisoned_training_set(is_train=True, args=visual_args)
        dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=visual_args)
        
        print(f"Dataset loaded: {args.dataset}")
        print(f"Clean validation set size: {len(dataset_val_clean)}")
        print(f"Poisoned validation set size: {len(dataset_val_poisoned)}")
        
        # 可视化多个样本
        print("\nVisualizing multiple samples...")
        visualize_samples(dataset_val_clean, dataset_val_poisoned, args.dataset, args.trigger_label, args.num_samples)
        
        # 可视化单个样本
        print("\nVisualizing single sample comparison...")
        visualize_single_comparison(dataset_val_clean, dataset_val_poisoned, args.dataset, args.trigger_label, index=0)
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()