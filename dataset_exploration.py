"""
Breast MRI Tumor Colorized Classification Dataset - Exploration Script
This script downloads and explores the dataset for research purposes.
"""

import kagglehub
import os
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Download dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("shuvokumarbasakbd/breast-mri-tumor-colorized-classification-dataset")
print(f"Path to dataset files: {path}")

# Dataset structure exploration
def explore_dataset_structure(dataset_path: str) -> Dict:
    """
    Explore the structure of the dataset.
    Returns a dictionary with dataset information.
    """
    dataset_info = {
        'root_path': dataset_path,
        'directories': [],
        'files': [],
        'image_count': 0,
        'augmentation_techniques': []
    }
    
    root = Path(dataset_path)
    
    # List all directories and files
    for item in root.iterdir():
        if item.is_dir():
            dataset_info['directories'].append(str(item.name))
            # Count images in subdirectories
            image_files = list(item.glob('**/*.png')) + list(item.glob('**/*.jpg')) + list(item.glob('**/*.jpeg'))
            dataset_info['image_count'] += len(image_files)
        elif item.is_file():
            dataset_info['files'].append(str(item.name))
    
    # Try to identify augmentation technique folders
    # Based on the 53 techniques mentioned
    augmentation_keywords = [
        'color', 'histogram', 'contrast', 'gaussian', 'edge', 'gamma',
        'blur', 'noise', 'window', 'mri', 'ct', 'thermal', 'vessel',
        'lesion', 'brightness', 'hue', 'pca', 'sobel', 'gradient'
    ]
    
    for dir_name in dataset_info['directories']:
        if any(keyword in dir_name.lower() for keyword in augmentation_keywords):
            dataset_info['augmentation_techniques'].append(dir_name)
    
    return dataset_info

# Analyze dataset statistics
def analyze_dataset_statistics(dataset_path: str, sample_size: int = 100) -> Dict:
    """
    Analyze basic statistics of the dataset.
    """
    stats = {
        'total_images': 0,
        'image_sizes': [],
        'mean_intensity': [],
        'std_intensity': [],
        'class_distribution': {}
    }
    
    root = Path(dataset_path)
    image_files = list(root.glob('**/*.png')) + list(root.glob('**/*.jpg')) + list(root.glob('**/*.jpeg'))
    
    stats['total_images'] = len(image_files)
    
    # Sample images for statistics
    sample_files = np.random.choice(image_files, min(sample_size, len(image_files)), replace=False)
    
    for img_path in sample_files:
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            stats['image_sizes'].append(img_array.shape)
            
            if len(img_array.shape) == 3:  # Color image
                stats['mean_intensity'].append(img_array.mean())
                stats['std_intensity'].append(img_array.std())
            else:  # Grayscale
                stats['mean_intensity'].append(img_array.mean())
                stats['std_intensity'].append(img_array.std())
            
            # Try to infer class from path
            path_parts = str(img_path).split(os.sep)
            for part in path_parts:
                if 'benign' in part.lower() or 'malignant' in part.lower() or 'normal' in part.lower():
                    class_name = part.lower()
                    stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1
                    break
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return stats

# Visualize sample images
def visualize_samples(dataset_path: str, num_samples: int = 9, save_path: str = 'sample_images.png'):
    """
    Visualize sample images from the dataset.
    """
    root = Path(dataset_path)
    image_files = list(root.glob('**/*.png')) + list(root.glob('**/*.jpg')) + list(root.glob('**/*.jpeg'))
    
    if len(image_files) == 0:
        print("No images found in dataset!")
        return
    
    sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, img_path in enumerate(sample_files):
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"{img_path.parent.name}\n{img_path.name[:30]}...", fontsize=8)
            axes[idx].axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample images saved to {save_path}")
    plt.close()

# Main exploration
if __name__ == "__main__":
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50 + "\n")
    
    # Explore structure
    print("1. Exploring dataset structure...")
    dataset_info = explore_dataset_structure(path)
    
    print(f"\nDataset Root: {dataset_info['root_path']}")
    print(f"Total Directories: {len(dataset_info['directories'])}")
    print(f"Total Files: {len(dataset_info['files'])}")
    print(f"Estimated Image Count: {dataset_info['image_count']}")
    print(f"\nDirectories found:")
    for dir_name in dataset_info['directories'][:10]:  # Show first 10
        print(f"  - {dir_name}")
    if len(dataset_info['directories']) > 10:
        print(f"  ... and {len(dataset_info['directories']) - 10} more")
    
    print(f"\nAugmentation Techniques Identified: {len(dataset_info['augmentation_techniques'])}")
    for tech in dataset_info['augmentation_techniques'][:10]:
        print(f"  - {tech}")
    
    # Analyze statistics
    print("\n2. Analyzing dataset statistics...")
    stats = analyze_dataset_statistics(path, sample_size=100)
    
    print(f"\nTotal Images Found: {stats['total_images']}")
    if stats['image_sizes']:
        unique_sizes = set(stats['image_sizes'])
        print(f"Unique Image Sizes: {len(unique_sizes)}")
        print(f"Sample Sizes: {list(unique_sizes)[:5]}")
    
    if stats['mean_intensity']:
        print(f"\nIntensity Statistics (sample):")
        print(f"  Mean: {np.mean(stats['mean_intensity']):.2f}")
        print(f"  Std: {np.mean(stats['std_intensity']):.2f}")
    
    if stats['class_distribution']:
        print(f"\nClass Distribution (inferred from paths):")
        for class_name, count in stats['class_distribution'].items():
            print(f"  {class_name}: {count}")
    
    # Visualize samples
    print("\n3. Creating sample visualizations...")
    visualize_samples(path, num_samples=9, save_path='dataset_samples.png')
    
    # Save exploration results
    print("\n4. Saving exploration results...")
    results = {
        'dataset_info': dataset_info,
        'statistics': {
            'total_images': stats['total_images'],
            'mean_intensity': float(np.mean(stats['mean_intensity'])) if stats['mean_intensity'] else None,
            'std_intensity': float(np.mean(stats['std_intensity'])) if stats['std_intensity'] else None,
            'class_distribution': stats['class_distribution']
        }
    }
    
    with open('dataset_exploration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("Exploration complete! Results saved to 'dataset_exploration_results.json'")
    print("Sample images saved to 'dataset_samples.png'")
    print("="*50)

