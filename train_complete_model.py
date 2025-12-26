"""
Complete Training Script for Breast MRI Tumor Classification
Comprehensive training, testing, and analysis with research-grade visualizations
Optimized for RTX 3050 6GB GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    cohen_kappa_score, matthews_corrcoef, hamming_loss,
    jaccard_score, top_k_accuracy_score
)
from scipy import stats
from scipy.stats import chi2_contingency
import matplotlib.patches as mpatches
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Configuration
CONFIG = {
    'dataset_path': 'breast_mri_dataset',
    'results_dir': 'training_results',
    'batch_size': 16,  # Optimized for RTX 3050 6GB
    'num_epochs': 100,  # Extended training
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'patience': 15,  # Early stopping patience
    'image_size': 224,
    'num_workers': 4,
    'device': None,  # Will be set automatically
    'gradient_accumulation_steps': 2,  # Effective batch size = 32
    'save_best_model': True,
    'use_mixed_precision': True,  # For memory efficiency
}

# Create results directory
results_dir = Path(CONFIG['results_dir'])
results_dir.mkdir(exist_ok=True)
(results_dir / 'models').mkdir(exist_ok=True)
(results_dir / 'plots').mkdir(exist_ok=True)
(results_dir / 'tables').mkdir(exist_ok=True)
(results_dir / 'checkpoints').mkdir(exist_ok=True)

print(f"Results will be saved to: {results_dir}")


class BreastMRIDataset(Dataset):
    """Dataset class for Breast MRI images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)
            return image, label


def load_dataset(dataset_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Load and split dataset into train/val/test"""
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    dataset_path = Path(dataset_path)
    image_paths = []
    labels = []
    class_names = []
    
    # Get all classes (Benign, Malignant)
    for class_dir in sorted(dataset_path.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Get images from all augmentation folders
            for aug_dir in class_dir.iterdir():
                if aug_dir.is_dir():
                    images = list(aug_dir.glob('*.jpg')) + list(aug_dir.glob('*.png')) + list(aug_dir.glob('*.jpeg'))
                    for img_path in images:
                        image_paths.append(img_path)
                        labels.append(len(class_names) - 1)  # 0 for Benign, 1 for Malignant
    
    print(f"Total images found: {len(image_paths)}")
    print(f"Classes: {class_names}")
    print(f"Class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {class_names[cls]}: {count} images ({count/len(labels)*100:.2f}%)")
    
    # Split dataset
    dataset_size = len(image_paths)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    indices = np.random.permutation(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"\nDataset Split:")
    print(f"  Training: {len(train_paths)} images ({len(train_paths)/dataset_size*100:.1f}%)")
    print(f"  Validation: {len(val_paths)} images ({len(val_paths)/dataset_size*100:.1f}%)")
    print(f"  Testing: {len(test_paths)} images ({len(test_paths)/dataset_size*100:.1f}%)")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_names


def create_data_loaders(train_data, val_data, test_data, config):
    """Create data loaders with appropriate transforms"""
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation and test
    val_test_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = BreastMRIDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = BreastMRIDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = BreastMRIDataset(test_paths, test_labels, transform=val_test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_model(num_classes=2):
    """Create ResNet model with transfer learning"""
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, config, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)
        
        if config['use_mixed_precision'] and scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / config['gradient_accumulation_steps']
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / config['gradient_accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        running_loss += loss.item() * config['gradient_accumulation_steps']
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs


def train_model(model, train_loader, val_loader, config, class_names):
    """Complete training loop"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    scaler = torch.cuda.amp.GradScaler() if config['use_mixed_precision'] else None
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config, scaler)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Print LR change if it happened
        if epoch > 0 and history['learning_rate'][-1] != history['learning_rate'][-2]:
            print(f"  → Learning rate reduced to {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            if config['save_best_model']:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, results_dir / 'models' / 'best_model.pth')
            print(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, results_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Memory cleanup
        if config['device'] == 'cuda':
            torch.cuda.empty_cache()
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history, best_val_acc


def evaluate_model(model, test_loader, device, class_names):
    """Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # ROC curve
    all_probs_array = np.array(all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs_array[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs_array[:, 1])
    pr_auc = average_precision_score(all_labels, all_probs_array[:, 1])
    
    # Additional metrics
    cohen_kappa = cohen_kappa_score(all_labels, all_preds)
    matthews_corr = matthews_corrcoef(all_labels, all_preds)
    hamming = hamming_loss(all_labels, all_preds)
    jaccard = jaccard_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'cohen_kappa': cohen_kappa,
        'matthews_corr': matthews_corr,
        'hamming_loss': hamming,
        'jaccard_score': jaccard,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
    }
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall: {recall_per_class[i]:.4f}")
        print(f"    F1-Score: {f1_per_class[i]:.4f}")
    
    print(f"\nAdditional Metrics:")
    print(f"  Cohen's Kappa: {cohen_kappa:.4f}")
    print(f"  Matthews Correlation: {matthews_corr:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  Jaccard Score: {jaccard:.4f}")
    
    return metrics, all_preds, all_labels, all_probs_array, fpr, tpr, precision_curve, recall_curve


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rate'], color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss vs Accuracy
    axes[1, 1].scatter(history['val_loss'], history['val_acc'], 
                       c=range(len(history['val_loss'])), cmap='viridis', s=50, alpha=0.6)
    axes[1, 1].set_xlabel('Validation Loss')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Loss vs Accuracy Relationship', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Epoch')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plot ROC curve"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_precision_recall_curve(precision_curve, recall_curve, pr_auc, save_path):
    """Plot Precision-Recall curve"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall_curve, precision_curve, color='darkblue', lw=2,
            label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(metrics, class_names, save_path):
    """Plot metrics comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics_data = {
        'Precision': metrics['precision_per_class'],
        'Recall': metrics['recall_per_class'],
        'F1-Score': metrics['f1_per_class']
    }
    
    x = np.arange(len(class_names))
    width = 0.6
    
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        bars = axes[idx].bar(x, values, width, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[idx].set_ylabel(metric_name, fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{metric_name} by Class', fontsize=13, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(class_names, fontsize=11)
        axes[idx].set_ylim([0, 1.1])
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_dataset_statistics(train_data, val_data, test_data, class_names, save_path):
    """Plot dataset statistics"""
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Split distribution
    splits = ['Train', 'Validation', 'Test']
    sizes = [len(train_paths), len(val_paths), len(test_paths)]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[0, 0].pie(sizes, labels=splits, autopct='%1.1f%%', startangle=90,
                   colors=colors, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0, 0].set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
    
    # Class distribution in train set
    train_counts = [train_labels.count(i) for i in range(len(class_names))]
    axes[0, 1].bar(class_names, train_counts, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
    axes[0, 1].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(train_counts):
        axes[0, 1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Class distribution across all splits
    all_labels = train_labels + val_labels + test_labels
    all_counts = [all_labels.count(i) for i in range(len(class_names))]
    axes[1, 0].bar(class_names, all_counts, color=['#9b59b6', '#f39c12'], alpha=0.8, edgecolor='black')
    axes[1, 0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(all_counts):
        axes[1, 0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Split comparison by class
    train_class_counts = [train_labels.count(i) for i in range(len(class_names))]
    val_class_counts = [val_labels.count(i) for i in range(len(class_names))]
    test_class_counts = [test_labels.count(i) for i in range(len(class_names))]
    
    x = np.arange(len(class_names))
    width = 0.25
    axes[1, 1].bar(x - width, train_class_counts, width, label='Train', alpha=0.8, edgecolor='black')
    axes[1, 1].bar(x, val_class_counts, width, label='Validation', alpha=0.8, edgecolor='black')
    axes[1, 1].bar(x + width, test_class_counts, width, label='Test', alpha=0.8, edgecolor='black')
    axes[1, 1].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_prediction_distribution(all_probs, class_names, save_path):
    """Plot prediction probability distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, class_name in enumerate(class_names):
        axes[0].hist(all_probs[:, i], bins=50, alpha=0.7, label=class_name, edgecolor='black')
    axes[0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confidence distribution
    max_probs = np.max(all_probs, axis=1)
    axes[1].hist(max_probs, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Maximum Prediction Probability (Confidence)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_metrics_table(metrics, class_names, save_path):
    """Create metrics table"""
    data = {
        'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)',
                   'ROC-AUC', 'PR-AUC'],
        'Value': [
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['pr_auc']:.4f}"
        ]
    }
    df_overall = pd.DataFrame(data)
    
    data_per_class = {
        'Class': class_names,
        'Precision': [f"{p:.4f}" for p in metrics['precision_per_class']],
        'Recall': [f"{r:.4f}" for r in metrics['recall_per_class']],
        'F1-Score': [f"{f:.4f}" for f in metrics['f1_per_class']]
    }
    df_per_class = pd.DataFrame(data_per_class)
    
    # Save as CSV
    df_overall.to_csv(results_dir / 'tables' / 'overall_metrics.csv', index=False)
    df_per_class.to_csv(results_dir / 'tables' / 'per_class_metrics.csv', index=False)
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Metric', 'Value'])
    for _, row in df_overall.iterrows():
        table_data.append([row['Metric'], row['Value']])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    for i in range(len(table_data)):
        if i == 0:
            table[(i, 0)].set_facecolor('#3498db')
            table[(i, 1)].set_facecolor('#3498db')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        else:
            table[(i, 0)].set_facecolor('#ecf0f1')
            table[(i, 1)].set_facecolor('#ecf0f1')
    
    plt.title('Overall Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_learning_curves_detailed(history, save_path):
    """Detailed learning curve analysis"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss with moving average
    ma_window = min(5, len(history['train_loss']) // 4)
    train_loss_ma = pd.Series(history['train_loss']).rolling(window=ma_window).mean()
    val_loss_ma = pd.Series(history['val_loss']).rolling(window=ma_window).mean()
    
    axes[0, 0].plot(epochs, history['train_loss'], alpha=0.3, color='blue', label='Train Loss (raw)')
    axes[0, 0].plot(epochs, train_loss_ma, color='blue', linewidth=2, label=f'Train Loss (MA-{ma_window})')
    axes[0, 0].plot(epochs, history['val_loss'], alpha=0.3, color='red', label='Val Loss (raw)')
    axes[0, 0].plot(epochs, val_loss_ma, color='red', linewidth=2, label=f'Val Loss (MA-{ma_window})')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Loss Curves with Moving Average', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy with moving average
    train_acc_ma = pd.Series(history['train_acc']).rolling(window=ma_window).mean()
    val_acc_ma = pd.Series(history['val_acc']).rolling(window=ma_window).mean()
    
    axes[0, 1].plot(epochs, history['train_acc'], alpha=0.3, color='blue', label='Train Acc (raw)')
    axes[0, 1].plot(epochs, train_acc_ma, color='blue', linewidth=2, label=f'Train Acc (MA-{ma_window})')
    axes[0, 1].plot(epochs, history['val_acc'], alpha=0.3, color='red', label='Val Acc (raw)')
    axes[0, 1].plot(epochs, val_acc_ma, color='red', linewidth=2, label=f'Val Acc (MA-{ma_window})')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Accuracy Curves with Moving Average', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss difference (overfitting indicator)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 0].plot(epochs, loss_diff, color='purple', linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(epochs, 0, loss_diff, where=(loss_diff > 0), alpha=0.3, color='red', label='Overfitting')
    axes[1, 0].fill_between(epochs, 0, loss_diff, where=(loss_diff <= 0), alpha=0.3, color='green', label='Underfitting')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Val Loss - Train Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Overfitting/Underfitting Indicator', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy difference
    acc_diff = np.array(history['val_acc']) - np.array(history['train_acc'])
    axes[1, 1].plot(epochs, acc_diff, color='orange', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].fill_between(epochs, 0, acc_diff, where=(acc_diff > 0), alpha=0.3, color='green')
    axes[1, 1].fill_between(epochs, 0, acc_diff, where=(acc_diff <= 0), alpha=0.3, color='red')
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Val Acc - Train Acc', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Generalization Gap Analysis', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[2, 0].plot(epochs, history['learning_rate'], color='green', linewidth=2, marker='o', markersize=3)
    axes[2, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[2, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2, 0].set_yscale('log')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Training efficiency (accuracy per epoch)
    efficiency = np.array(history['val_acc']) / (np.array(history['val_loss']) + 1e-8)
    axes[2, 1].plot(epochs, efficiency, color='teal', linewidth=2)
    axes[2, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2, 1].set_ylabel('Efficiency (Acc/Loss)', fontsize=12, fontweight='bold')
    axes[2, 1].set_title('Training Efficiency Over Time', fontsize=14, fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix_detailed(cm, class_names, save_path):
    """Detailed confusion matrix with percentages"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Absolute values
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Absolute Values)', fontsize=14, fontweight='bold')
    
    # Percentage values
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Oranges', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5, linecolor='gray')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_performance_heatmap(metrics, class_names, save_path):
    """Performance metrics heatmap"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Per-class metrics
    metrics_data = np.array([
        metrics['precision_per_class'],
        metrics['recall_per_class'],
        metrics['f1_per_class']
    ])
    
    sns.heatmap(metrics_data, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[0],
                xticklabels=class_names, yticklabels=['Precision', 'Recall', 'F1-Score'],
                cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='gray')
    axes[0].set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    
    # Overall vs per-class comparison
    overall_metrics = np.array([
        [metrics['precision'], metrics['recall'], metrics['f1_score']],
        metrics_data.mean(axis=1)
    ])
    
    sns.heatmap(overall_metrics, annot=True, fmt='.4f', cmap='RdYlGn', ax=axes[1],
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=['Weighted Average', 'Macro Average'],
                cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='gray')
    axes[1].set_title('Overall vs Macro Average Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_analysis(all_labels, all_preds, all_probs, class_names, save_path):
    """Error analysis visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Error types
    correct = np.array(all_labels) == np.array(all_preds)
    errors = ~correct
    error_probs = all_probs[errors]
    error_labels = np.array(all_labels)[errors]
    error_preds = np.array(all_preds)[errors]
    
    # Confidence distribution for correct vs incorrect
    correct_probs = np.max(all_probs[correct], axis=1)
    incorrect_probs = np.max(all_probs[errors], axis=1)
    
    axes[0, 0].hist(correct_probs, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[0, 0].hist(incorrect_probs, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Model Confidence', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Confidence Distribution: Correct vs Incorrect', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error confusion (what was predicted vs what it should be)
    if len(error_labels) > 0:
        error_cm = confusion_matrix(error_labels, error_preds)
        if error_cm.size > 0:
            sns.heatmap(error_cm, annot=True, fmt='d', cmap='Reds', ax=axes[0, 1],
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={'label': 'Error Count'}, linewidths=0.5, linecolor='gray')
            axes[0, 1].set_xlabel('Predicted (Errors Only)', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('True Label (Errors Only)', fontsize=12, fontweight='bold')
            axes[0, 1].set_title('Error Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Error rate by class
    error_rate_by_class = []
    for i in range(len(class_names)):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            error_rate = np.sum((np.array(all_labels) == i) & errors) / np.sum(class_mask)
            error_rate_by_class.append(error_rate)
        else:
            error_rate_by_class.append(0)
    
    bars = axes[1, 0].bar(class_names, error_rate_by_class, color=['#e74c3c', '#3498db'], alpha=0.8, edgecolor='black')
    axes[1, 0].set_ylabel('Error Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Error Rate by Class', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, max(error_rate_by_class) * 1.2 if max(error_rate_by_class) > 0 else 0.1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(error_rate_by_class):
        axes[1, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Prediction probability for errors
    if len(error_probs) > 0:
        axes[1, 1].scatter(error_probs[:, 0], error_probs[:, 1], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[1, 1].set_xlabel(f'Probability: {class_names[0]}', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel(f'Probability: {class_names[1]}', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Error Probability Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].axline([0.5, 0.5], [1, 0], color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_comparison(fpr, tpr, roc_auc, save_path):
    """Enhanced ROC curve with additional metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Standard ROC
    axes[0].plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0].fill_between(fpr, 0, tpr, alpha=0.3, color='darkorange')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # ROC with optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = fpr[optimal_idx]
    axes[1].plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[1].plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12, 
                label=f'Optimal Point (FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('ROC Curve with Optimal Threshold', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_statistical_analysis(all_labels, all_preds, all_probs, class_names, save_path):
    """Statistical analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Class distribution in predictions vs actual
    actual_counts = [np.sum(np.array(all_labels) == i) for i in range(len(class_names))]
    pred_counts = [np.sum(np.array(all_preds) == i) for i in range(len(class_names))]
    
    x = np.arange(len(class_names))
    width = 0.35
    axes[0, 0].bar(x - width/2, actual_counts, width, label='Actual', alpha=0.8, edgecolor='black')
    axes[0, 0].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8, edgecolor='black')
    axes[0, 0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Actual vs Predicted Class Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(class_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Probability distribution by class
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        class_probs = all_probs[class_mask, i]
        axes[0, 1].hist(class_probs, bins=30, alpha=0.6, label=f'{class_name} (True)', edgecolor='black')
    axes[0, 1].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Probability Distribution by True Class', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot of probabilities by class
    prob_data = [all_probs[np.array(all_labels) == i, i] for i in range(len(class_names))]
    bp = axes[1, 0].boxplot(prob_data, labels=class_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Probability Distribution Box Plot by Class', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Correlation between predicted probabilities
    if len(class_names) == 2:
        axes[1, 1].scatter(all_probs[:, 0], all_probs[:, 1], alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        axes[1, 1].set_xlabel(f'Probability: {class_names[0]}', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel(f'Probability: {class_names[1]}', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Probability Correlation', fontsize=14, fontweight='bold')
        axes[1, 1].axline([0.5, 0.5], [1, 0], color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_comprehensive_metrics_table(metrics, class_names, history, save_path):
    """Create comprehensive metrics table"""
    # Overall metrics
    overall_data = {
        'Metric': [
            'Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)',
            'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)',
            'ROC-AUC', 'PR-AUC', 'Cohen\'s Kappa', 'Matthews Correlation'
        ],
        'Value': [
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            f"{np.mean(metrics['precision_per_class']):.4f}",
            f"{np.mean(metrics['recall_per_class']):.4f}",
            f"{np.mean(metrics['f1_per_class']):.4f}",
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['pr_auc']:.4f}",
            f"{metrics.get('cohen_kappa', 0):.4f}",
            f"{metrics.get('matthews_corr', 0):.4f}"
        ]
    }
    
    # Training metrics
    train_data = {
        'Metric': [
            'Best Train Accuracy', 'Best Train Loss', 'Final Train Accuracy', 'Final Train Loss',
            'Best Val Accuracy', 'Best Val Loss', 'Final Val Accuracy', 'Final Val Loss',
            'Total Epochs', 'Best Epoch', 'Final Learning Rate'
        ],
        'Value': [
            f"{max(history['train_acc']):.4f}",
            f"{min(history['train_loss']):.4f}",
            f"{history['train_acc'][-1]:.4f}",
            f"{history['train_loss'][-1]:.4f}",
            f"{max(history['val_acc']):.4f}",
            f"{min(history['val_loss']):.4f}",
            f"{history['val_acc'][-1]:.4f}",
            f"{history['val_loss'][-1]:.4f}",
            f"{len(history['train_acc'])}",
            f"{np.argmax(history['val_acc']) + 1}",
            f"{history['learning_rate'][-1]:.6f}"
        ]
    }
    
    # Per-class detailed
    per_class_data = {
        'Class': class_names,
        'Precision': [f"{p:.4f}" for p in metrics['precision_per_class']],
        'Recall': [f"{r:.4f}" for r in metrics['recall_per_class']],
        'F1-Score': [f"{f:.4f}" for f in metrics['f1_per_class']],
        'Support': [f"{int(np.sum(np.array(metrics['confusion_matrix'])[i, :]))}" 
                    for i in range(len(class_names))]
    }
    
    # Save as CSV
    pd.DataFrame(overall_data).to_csv(results_dir / 'tables' / 'comprehensive_overall_metrics.csv', index=False)
    pd.DataFrame(train_data).to_csv(results_dir / 'tables' / 'training_metrics.csv', index=False)
    pd.DataFrame(per_class_data).to_csv(results_dir / 'tables' / 'detailed_per_class_metrics.csv', index=False)
    
    # Create visual tables
    fig, axes = plt.subplots(3, 1, figsize=(14, 20))
    
    for idx, (data, title) in enumerate([
        (overall_data, 'Overall Performance Metrics'),
        (train_data, 'Training History Metrics'),
        (per_class_data, 'Per-Class Detailed Metrics')
    ]):
        ax = axes[idx]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [[k, v] for k, v in zip(data[list(data.keys())[0]], 
                                              data[list(data.keys())[1]])]
        table_data.insert(0, list(data.keys()))
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        for i in range(len(table_data)):
            if i == 0:
                for j in range(len(table_data[0])):
                    table[(i, j)].set_facecolor('#2c3e50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                for j in range(len(table_data[0])):
                    color = '#ecf0f1' if i % 2 == 0 else '#bdc3c7'
                    table[(i, j)].set_facecolor(color)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison_table(metrics, save_path):
    """Create model comparison table for future use"""
    comparison_data = {
        'Model': ['ResNet18 (Current)'],
        'Accuracy': [f"{metrics['accuracy']:.4f}"],
        'Precision': [f"{metrics['precision']:.4f}"],
        'Recall': [f"{metrics['recall']:.4f}"],
        'F1-Score': [f"{metrics['f1_score']:.4f}"],
        'ROC-AUC': [f"{metrics['roc_auc']:.4f}"],
        'PR-AUC': [f"{metrics['pr_auc']:.4f}"]
    }
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(results_dir / 'tables' / 'model_comparison.csv', index=False)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [df.columns.tolist()] + df.values.tolist()
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    for i in range(len(table_data)):
        if i == 0:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#16a085')
                table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Main training and evaluation pipeline"""
    print("="*60)
    print("BREAST MRI TUMOR CLASSIFICATION - COMPLETE TRAINING PIPELINE")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set device
    CONFIG['device'] = get_device()
    print(f"\nUsing device: {CONFIG['device']}")
    
    # Adjust settings for CPU if needed
    if CONFIG['device'] == 'cpu':
        CONFIG['batch_size'] = 8  # Smaller batch for CPU
        CONFIG['num_workers'] = 2  # Fewer workers for CPU
        CONFIG['use_mixed_precision'] = False  # Mixed precision only for GPU
        print("  Adjusted settings for CPU training (smaller batch size)")
    
    # Load dataset
    train_data, val_data, test_data, class_names = load_dataset(
        CONFIG['dataset_path'], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, CONFIG
    )
    
    # Plot dataset statistics
    plot_dataset_statistics(
        train_data, val_data, test_data, class_names,
        results_dir / 'plots' / '01_dataset_statistics.png'
    )
    
    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    model = create_model(num_classes=len(class_names))
    print(f"Model: ResNet18")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trained_model, history, best_val_acc = train_model(
        model, train_loader, val_loader, CONFIG, class_names
    )
    
    # Load best model for evaluation
    checkpoint = torch.load(results_dir / 'models' / 'best_model.pth')
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate on test set
    device = torch.device(CONFIG['device'])
    metrics, all_preds, all_labels, all_probs, fpr, tpr, precision_curve, recall_curve = evaluate_model(
        trained_model, test_loader, device, class_names
    )
    
    # Save metrics
    with open(results_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create all visualizations
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    # Basic visualizations
    print("\n1. Basic Training Visualizations...")
    plot_training_history(history, results_dir / 'plots' / '01_training_history.png')
    plot_learning_curves_detailed(history, results_dir / 'plots' / '02_detailed_learning_curves.png')
    
    # Confusion matrices
    print("\n2. Confusion Matrix Analysis...")
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']), class_names,
        results_dir / 'plots' / '03_confusion_matrix.png'
    )
    plot_confusion_matrix_detailed(
        np.array(metrics['confusion_matrix']), class_names,
        results_dir / 'plots' / '04_confusion_matrix_detailed.png'
    )
    
    # ROC and PR curves
    print("\n3. ROC and Precision-Recall Curves...")
    plot_roc_curve(fpr, tpr, metrics['roc_auc'], results_dir / 'plots' / '05_roc_curve.png')
    plot_roc_comparison(fpr, tpr, metrics['roc_auc'], results_dir / 'plots' / '06_roc_comparison.png')
    plot_precision_recall_curve(
        precision_curve, recall_curve, metrics['pr_auc'],
        results_dir / 'plots' / '07_precision_recall_curve.png'
    )
    
    # Performance metrics
    print("\n4. Performance Metrics Analysis...")
    plot_metrics_comparison(
        metrics, class_names, results_dir / 'plots' / '08_metrics_comparison.png'
    )
    plot_performance_heatmap(
        metrics, class_names, results_dir / 'plots' / '09_performance_heatmap.png'
    )
    
    # Prediction analysis
    print("\n5. Prediction Distribution Analysis...")
    plot_prediction_distribution(
        all_probs, class_names, results_dir / 'plots' / '10_prediction_distribution.png'
    )
    plot_statistical_analysis(
        all_labels, all_preds, all_probs, class_names,
        results_dir / 'plots' / '11_statistical_analysis.png'
    )
    
    # Error analysis
    print("\n6. Error Analysis...")
    plot_error_analysis(
        all_labels, all_preds, all_probs, class_names,
        results_dir / 'plots' / '12_error_analysis.png'
    )
    
    # Metrics tables
    print("\n7. Comprehensive Metrics Tables...")
    create_metrics_table(
        metrics, class_names, results_dir / 'plots' / '13_metrics_table.png'
    )
    create_comprehensive_metrics_table(
        metrics, class_names, history, results_dir / 'plots' / '14_comprehensive_metrics_table.png'
    )
    plot_model_comparison_table(
        metrics, results_dir / 'plots' / '15_model_comparison_table.png'
    )
    
    print("\n✓ All visualizations generated successfully!")
    
    # Save configuration
    config_save = CONFIG.copy()
    config_save['class_names'] = class_names
    config_save['best_val_acc'] = float(best_val_acc)
    config_save['training_completed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(results_dir / 'training_config.json', 'w') as f:
        json.dump(config_save, f, indent=2)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / 'training_history.csv', index=False)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAll results saved to: {results_dir}")
    print(f"  - Models: {results_dir / 'models'}")
    print(f"  - Plots: {results_dir / 'plots'} ({len(list((results_dir / 'plots').glob('*.png')))} visualizations)")
    print(f"  - Tables: {results_dir / 'tables'} ({len(list((results_dir / 'tables').glob('*.csv')))} CSV files)")
    print(f"  - Checkpoints: {results_dir / 'checkpoints'}")
    
    print("\n" + "="*60)
    print("GENERATED FILES SUMMARY")
    print("="*60)
    print("\nVisualizations (15 plots):")
    print("  01. Dataset Statistics")
    print("  02. Training History")
    print("  03. Detailed Learning Curves")
    print("  04. Confusion Matrix")
    print("  05. Confusion Matrix (Detailed)")
    print("  06. ROC Curve")
    print("  07. ROC Comparison")
    print("  08. Precision-Recall Curve")
    print("  09. Metrics Comparison")
    print("  10. Performance Heatmap")
    print("  11. Prediction Distribution")
    print("  12. Statistical Analysis")
    print("  13. Error Analysis")
    print("  14. Metrics Table")
    print("  15. Comprehensive Metrics Table")
    print("  16. Model Comparison Table")
    
    print("\nCSV Tables:")
    print("  - overall_metrics.csv")
    print("  - per_class_metrics.csv")
    print("  - comprehensive_overall_metrics.csv")
    print("  - training_metrics.csv")
    print("  - detailed_per_class_metrics.csv")
    print("  - model_comparison.csv")
    print("  - training_history.csv")
    
    print("\nJSON Files:")
    print("  - test_metrics.json")
    print("  - training_config.json")
    
    print("\nModel Files:")
    print("  - best_model.pth (Best trained model)")
    print("  - checkpoint_epoch_*.pth (Periodic checkpoints)")
    
    print("\n" + "="*60)
    print(f"FINAL TEST SET PERFORMANCE")
    print("="*60)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"  Matthews Correlation: {metrics['matthews_corr']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

