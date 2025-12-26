"""
Template for Systematic Evaluation of 53 Augmentation Techniques
Research Direction #1 from RESEARCH_PROPOSAL.md

This script provides a framework for evaluating the impact of different
augmentation techniques on breast MRI tumor classification performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the 53 augmentation techniques (as mentioned in the dataset description)
AUGMENTATION_TECHNIQUES = [
    # 1-13: Original visualization methods
    "Basic_Color_Map", "Adaptive_Histogram_Equalization", "Contrast_Stretching",
    "Gaussian_Blur", "Edge_Detection", "Random_Color_Palette", "Gamma_Correction",
    "LUT_Color_Map", "Alpha_Blending", "Render_3D", "Heatmap_Visualization",
    "Volume_Render_3D", "Interactive_Segmentation",
    
    # 14-33: CT/MRI-specific techniques
    "CT_Soft_Tissue", "CT_Bone", "CT_Lung", "CT_Brain", "MRI_T1", "MRI_T2",
    "MRI_FLAIR", "CT_Window_Sweep", "Histogram_Match_MRI", "Zscore_Normalization",
    "Rician_Noise", "Poisson_Noise", "Gaussian_Noise", "Speckle_Noise",
    "Motion_Blur", "Gibbs_Ringing", "Laplacian_Sharpen", "Vesselness_Frangi",
    "CT_Window_Heatmap", "Lesion_Highlight",
    
    # 34-53: General augmentation techniques
    "Brightness_Jitter", "Contrast_Jitter", "Color_Temperature", "HSV_Jitter",
    "Channel_Shuffle", "Random_Inversion", "PCA_Color_Aug", "Sobel_Overlay",
    "Color_Tint", "Mixup_Noise", "Solarize", "Posterize", "Hue_Shift",
    "Color_Spread", "Gradient_Map", "Edge_ColorBlend", "Thermal_LUT",
    "Random_Contrast_LUT", "Local_Color_Normalization", "Random_Pepper_Color"
]

class BreastMRIDataset(Dataset):
    """
    Dataset class for loading breast MRI images.
    Assumes dataset structure: dataset_path/augmentation_technique/class/image.png
    """
    def __init__(self, dataset_path: str, augmentation_technique: str, 
                 split: str = 'train', transform: Optional[transforms.Compose] = None):
        self.dataset_path = Path(dataset_path)
        self.augmentation_technique = augmentation_technique
        self.split = split
        self.transform = transform
        
        # Load images and labels
        self.images = []
        self.labels = []
        
        aug_path = self.dataset_path / augmentation_technique
        if not aug_path.exists():
            print(f"Warning: {augmentation_technique} directory not found!")
            return
        
        # Assume class folders (e.g., 'benign', 'malignant', 'normal')
        for class_folder in aug_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                class_idx = self._get_class_index(class_name)
                
                # Load images from this class
                image_files = list(class_folder.glob('*.png')) + \
                             list(class_folder.glob('*.jpg')) + \
                             list(class_folder.glob('*.jpeg'))
                
                for img_path in image_files:
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images for {augmentation_technique} ({split})")
    
    def _get_class_index(self, class_name: str) -> int:
        """Map class name to index"""
        class_mapping = {
            'benign': 0,
            'malignant': 1,
            'normal': 2
        }
        class_name_lower = class_name.lower()
        for key, value in class_mapping.items():
            if key in class_name_lower:
                return value
        return 0  # Default
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

class AugmentationEvaluator:
    """
    Evaluates the impact of different augmentation techniques on model performance.
    """
    def __init__(self, dataset_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.dataset_path = dataset_path
        self.device = device
        self.results = {}
    
    def train_model(self, augmentation_technique: str, epochs: int = 10, 
                   batch_size: int = 32, learning_rate: float = 0.001) -> Dict:
        """
        Train a model on images from a specific augmentation technique.
        Returns training history and final metrics.
        """
        print(f"\nTraining model for: {augmentation_technique}")
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets (assuming 80/20 split)
        full_dataset = BreastMRIDataset(self.dataset_path, augmentation_technique, 
                                       transform=train_transform)
        
        if len(full_dataset) == 0:
            return {'error': 'No data found for this augmentation technique'}
        
        # Simple train/val split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Update val dataset transform
        val_dataset.dataset.transform = val_transform
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Model (using ResNet18 as baseline)
        model = models.resnet18(pretrained=True)
        num_classes = len(set(full_dataset.labels))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = accuracy_score(all_labels, all_preds)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Final metrics
        final_metrics = {
            'final_val_accuracy': val_acc,
            'final_val_loss': val_loss,
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1_score': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        return {
            'history': history,
            'metrics': final_metrics,
            'num_samples': len(full_dataset)
        }
    
    def evaluate_all_techniques(self, techniques: List[str] = None, 
                               epochs: int = 10, batch_size: int = 32) -> pd.DataFrame:
        """
        Evaluate all augmentation techniques and return comparison results.
        """
        if techniques is None:
            techniques = AUGMENTATION_TECHNIQUES
        
        results_list = []
        
        for technique in techniques:
            try:
                result = self.train_model(technique, epochs=epochs, batch_size=batch_size)
                
                if 'error' not in result:
                    results_list.append({
                        'augmentation_technique': technique,
                        'val_accuracy': result['metrics']['final_val_accuracy'],
                        'val_loss': result['metrics']['final_val_loss'],
                        'precision': result['metrics']['precision'],
                        'recall': result['metrics']['recall'],
                        'f1_score': result['metrics']['f1_score'],
                        'num_samples': result['num_samples']
                    })
            except Exception as e:
                print(f"Error evaluating {technique}: {e}")
                continue
        
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('val_accuracy', ascending=False)
        
        return results_df
    
    def visualize_results(self, results_df: pd.DataFrame, save_path: str = 'augmentation_comparison.png'):
        """
        Visualize comparison of augmentation techniques.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top techniques by accuracy
        top_10 = results_df.head(10)
        axes[0, 0].barh(range(len(top_10)), top_10['val_accuracy'].values)
        axes[0, 0].set_yticks(range(len(top_10)))
        axes[0, 0].set_yticklabels(top_10['augmentation_technique'].values, fontsize=8)
        axes[0, 0].set_xlabel('Validation Accuracy')
        axes[0, 0].set_title('Top 10 Augmentation Techniques by Accuracy')
        axes[0, 0].invert_yaxis()
        
        # Accuracy distribution
        axes[0, 1].hist(results_df['val_accuracy'], bins=20, edgecolor='black')
        axes[0, 1].set_xlabel('Validation Accuracy')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Validation Accuracies')
        
        # F1 Score vs Accuracy
        axes[1, 0].scatter(results_df['val_accuracy'], results_df['f1_score'], alpha=0.6)
        axes[1, 0].set_xlabel('Validation Accuracy')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Accuracy vs F1 Score')
        
        # Sample size vs Performance
        axes[1, 1].scatter(results_df['num_samples'], results_df['val_accuracy'], alpha=0.6)
        axes[1, 1].set_xlabel('Number of Samples')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].set_title('Sample Size vs Performance')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results visualization saved to {save_path}")
        plt.close()

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    DATASET_PATH = "path/to/dataset"  # Update with actual path
    
    evaluator = AugmentationEvaluator(DATASET_PATH)
    
    # Evaluate a subset of techniques first (for testing)
    test_techniques = AUGMENTATION_TECHNIQUES[:5]  # First 5 techniques
    
    print("Starting systematic evaluation...")
    results_df = evaluator.evaluate_all_techniques(
        techniques=test_techniques,
        epochs=5,  # Reduce for testing
        batch_size=16
    )
    
    # Save results
    results_df.to_csv('augmentation_evaluation_results.csv', index=False)
    print("\nResults saved to 'augmentation_evaluation_results.csv'")
    print("\nTop 5 Techniques:")
    print(results_df.head(5))
    
    # Visualize
    evaluator.visualize_results(results_df)
    
    print("\nEvaluation complete!")

