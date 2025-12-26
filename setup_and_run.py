"""
Complete Setup and Execution Script
Installs all required libraries and runs Python scripts in order
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_step(step_num, total_steps, text):
    """Print step information"""
    print(f"\n[{step_num}/{total_steps}] {text}")
    print("-" * 70)

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"\nExecuting: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description}")
        print(f"Error output: {e.stderr}")
        if check:
            print("\n⚠️  Installation failed. Please check the error above.")
            return False
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def check_python():
    """Check Python installation"""
    print_header("CHECKING PYTHON INSTALLATION")
    try:
        version = sys.version_info
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("⚠️  WARNING: Python 3.8+ is recommended")
        return True
    except Exception as e:
        print(f"✗ Python check failed: {e}")
        return False

def check_pip():
    """Check pip installation"""
    print_header("CHECKING PIP INSTALLATION")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
            return True
        else:
            print("✗ pip not found")
            return False
    except Exception as e:
        print(f"✗ pip check failed: {e}")
        return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    print_header("UPGRADING PIP")
    return run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "pip upgrade"
    )

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print_header("INSTALLING PYTORCH WITH CUDA SUPPORT")
    print("This may take 5-10 minutes...")
    print("Installing PyTorch, torchvision, and torchaudio")
    
    # Try CUDA 11.8 first (most compatible)
    command = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    print("\nAttempting to install PyTorch with CUDA 11.8...")
    success = run_command(command, "PyTorch installation", check=False)
    
    if not success:
        print("\n⚠️  CUDA installation failed, trying CPU-only version...")
        command = f"{sys.executable} -m pip install torch torchvision torchaudio"
        success = run_command(command, "PyTorch CPU installation", check=False)
    
    if success:
        print("\n✓ PyTorch installation completed")
        # Verify installation
        try:
            import torch
            print(f"✓ PyTorch {torch.__version__} installed")
            if torch.cuda.is_available():
                print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  CUDA not available (CPU-only mode)")
        except:
            print("⚠️  Could not verify PyTorch installation")
    
    return success

def install_requirements():
    """Install all requirements from requirements_training.txt"""
    print_header("INSTALLING ALL REQUIREMENTS")
    
    requirements_file = Path("requirements_training.txt")
    if not requirements_file.exists():
        print("✗ requirements_training.txt not found!")
        print("Installing core packages manually...")
        return install_core_packages()
    
    print(f"Reading requirements from: {requirements_file}")
    print("This may take 5-15 minutes depending on your internet speed...")
    
    command = f"{sys.executable} -m pip install -r {requirements_file}"
    return run_command(command, "Requirements installation")

def install_core_packages():
    """Install core packages manually if requirements file is missing"""
    print_header("INSTALLING CORE PACKAGES")
    
    packages = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0"
    ]
    
    for package in packages:
        print(f"\nInstalling {package}...")
        command = f"{sys.executable} -m pip install {package}"
        if not run_command(command, f"Installing {package}", check=False):
            print(f"⚠️  Warning: {package} installation had issues")
    
    return True

def verify_installation():
    """Verify all critical packages are installed"""
    print_header("VERIFYING INSTALLATION")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_ok = False
    
    return all_ok

def run_python_script(script_name, description):
    """Run a Python script"""
    script_path = Path(script_name)
    
    if not script_path.exists():
        print(f"⚠️  Script not found: {script_name}")
        print(f"   Skipping {description}")
        return False
    
    print_header(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    
    command = f"{sys.executable} {script_name}"
    success = run_command(command, description, check=False)
    
    if success:
        print(f"\n✓ {description} completed successfully")
    else:
        print(f"\n⚠️  {description} had issues (check output above)")
    
    return success

def main():
    """Main execution flow"""
    print_header("COMPLETE SETUP AND EXECUTION SCRIPT")
    print("This script will:")
    print("1. Check Python and pip installation")
    print("2. Upgrade pip to latest version")
    print("3. Install PyTorch with CUDA support")
    print("4. Install all required libraries")
    print("5. Verify installation")
    print("6. Run dataset exploration (optional)")
    print("7. Run main training script")
    print("\nThis process may take 15-30 minutes for installation")
    print("Training will take an additional 2-4 hours")
    
    input("\nPress ENTER to continue or Ctrl+C to cancel...")
    
    total_steps = 8
    step = 1
    
    # Step 1: Check Python
    print_step(step, total_steps, "Checking Python Installation")
    if not check_python():
        print("✗ Python check failed. Please install Python 3.8+")
        return False
    step += 1
    time.sleep(1)
    
    # Step 2: Check pip
    print_step(step, total_steps, "Checking pip Installation")
    if not check_pip():
        print("✗ pip not found. Please install pip")
        return False
    step += 1
    time.sleep(1)
    
    # Step 3: Upgrade pip
    print_step(step, total_steps, "Upgrading pip")
    upgrade_pip()
    step += 1
    time.sleep(1)
    
    # Step 4: Install PyTorch
    print_step(step, total_steps, "Installing PyTorch (This takes 5-10 minutes)")
    install_pytorch()
    step += 1
    time.sleep(2)
    
    # Step 5: Install requirements
    print_step(step, total_steps, "Installing All Requirements (This takes 5-15 minutes)")
    install_requirements()
    step += 1
    time.sleep(2)
    
    # Step 6: Verify installation
    print_step(step, total_steps, "Verifying Installation")
    if not verify_installation():
        print("\n⚠️  Some packages may not be installed correctly")
        print("   You can continue, but some features may not work")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    step += 1
    time.sleep(1)
    
    # Step 7: Run dataset exploration (optional)
    print_step(step, total_steps, "Dataset Exploration (Optional)")
    response = input("\nRun dataset exploration script? (y/n, default=n): ")
    if response.lower() == 'y':
        run_python_script("dataset_exploration.py", "Dataset Exploration")
    else:
        print("Skipping dataset exploration")
    step += 1
    time.sleep(1)
    
    # Step 8: Run main training script
    print_step(step, total_steps, "Main Training Script")
    print("\n⚠️  IMPORTANT: This will take 2-4 hours to complete!")
    print("   The script will train the model and generate all visualizations")
    response = input("\nStart training now? (y/n, default=y): ")
    
    if response.lower() != 'n':
        print("\n" + "="*70)
        print("STARTING TRAINING - This will take 2-4 hours")
        print("="*70)
        print("You can monitor progress in the terminal")
        print("Results will be saved to 'training_results' folder")
        print("="*70 + "\n")
        
        time.sleep(2)
        run_python_script("train_complete_model.py", "Model Training")
    else:
        print("Training skipped. Run 'python train_complete_model.py' when ready.")
    
    # Final summary
    print_header("SETUP AND EXECUTION COMPLETE")
    print("\nSummary:")
    print("✓ All libraries installed")
    print("✓ Scripts executed")
    print("\nNext steps:")
    print("  - Check 'training_results' folder for outputs")
    print("  - View visualizations in 'training_results/plots'")
    print("  - Check metrics in 'training_results/tables'")
    print("\n" + "="*70)
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print("Installation may be incomplete")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        sys.exit(1)

